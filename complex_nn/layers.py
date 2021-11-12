import haiku as hk
from haiku import RNNCore, LSTMState
from haiku._src import utils
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from haiku._src.utils import get_channel_index
from functools import partial

from typing import Optional, Tuple, Union, Sequence
import warnings

from complex_nn.initializers import CmplxRndUniform, CmplxTruncatedNormal, Cmplx_Xavier_Init, Cmplx_He_Init




##################### LINEAR LAYERS ##################################################


class Cmplx_Linear(hk.Module):
  """Linear module."""

  def __init__(
      self,
      output_size: int,
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    """Constructs the Linear module.
    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses complex variant
              of Xavier initialization with the Rayleigh distribution.
      b_init: Optional initializer for bias. By default, uniform in [-0.001, 0.001].
      name: Name of the module.
    """
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or CmplxRndUniform(minval=-0.001, maxval=0.001)

  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      precision: Optional[jax.lax.Precision] = None,
  ) -> jnp.ndarray:
    """Computes a linear transform of the input."""
    
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      w_init = Cmplx_Xavier_Init(input_size, output_size)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

    out = jnp.dot(inputs, w, precision=precision)

    if self.with_bias:
      b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    return out



############# DROPOUT ####################################################################



class Dropout(hk.Module):
  """Basic implementation of a Dropout layer."""

  def __init__(
      self,
      rate: float,
      name: Optional[str] = None,
  ):
    """Constructs the Dropout module.
    Args:
      rate: Probability that each element of x is discarded. Must be a scalar in the range [0, 1).
      name: Name of the module.
    """
    super().__init__(name=name)
    self.rate = rate
    
  def __call__(
      self,
      x: jnp.ndarray,
      is_training: Optional[bool] = True,
  ) -> jnp.ndarray:
    """Wrapper layer of the function hk.Dropout."""

    if is_training:
      return hk.dropout(hk.next_rng_key(), self.rate, x)
    else:
      return x





##################### CONVOLUTIONAL LAYERS ###########################################


class Cmplx_ConvND(hk.Module):
  """General N-dimensional complex convolutional."""

  def __init__(
      self,
      num_spatial_dims: int,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      rate: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]], hk.pad.PadFn,
                     Sequence[hk.pad.PadFn]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "channels_last",
      mask: Optional[jnp.ndarray] = None,
      feature_group_count: int = 1,
      name: Optional[str] = None,
  ):
    """Initializes the module.
    Args:
      num_spatial_dims: The number of spatial dimensions of the input.
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length ``num_spatial_dims``. 1 corresponds to standard ND convolution,
        ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or a
        sequence of n ``(low, high)`` integer pairs that give the padding to
        apply before and after each spatial dimension. or a callable or sequence
        of callables of size ``num_spatial_dims``. Any callables must take a
        single integer argument equal to the effective kernel size and return a
        sequence of two integers representing the padding before and after. See
        ``haiku.pad.*`` for more details and example functions. Defaults to
        ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input.  Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default, ``channels_last``.
      mask: Optional mask of the weights.
      feature_group_count: Optional number of groups in group convolution.
        Default value of 1 corresponds to normal dense convolution. If a higher
        value is used, convolutions are applied separately to that many groups,
        then stacked together. This reduces the number of parameters
        and possibly the compute for a given ``output_channels``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      name: The name of the module.
    """
    super().__init__(name=name)
    if num_spatial_dims <= 0:
      raise ValueError(
          "We only support convolution operations for `num_spatial_dims` "
          f"greater than 0, received num_spatial_dims={num_spatial_dims}.")

    self.num_spatial_dims = num_spatial_dims
    self.output_channels = output_channels
    self.kernel_shape = (utils.replicate(kernel_shape, num_spatial_dims, "kernel_shape"))
    self.with_bias = with_bias
    self.stride = utils.replicate(stride, num_spatial_dims, "strides")
    self.w_init = w_init
    self.b_init = b_init or CmplxRndUniform(minval=-0.001, maxval=0.001)
    self.mask = mask
    self.feature_group_count = feature_group_count
    self.lhs_dilation = utils.replicate(1, num_spatial_dims, "lhs_dilation")
    self.kernel_dilation = (utils.replicate(rate, num_spatial_dims, "kernel_dilation"))
    self.data_format = data_format
    self.channel_index = utils.get_channel_index(data_format)
    self.dimension_numbers = to_dimension_numbers(
        num_spatial_dims, channels_last=(self.channel_index == -1),
        transpose=False)

    if isinstance(padding, str):
      self.padding = padding.upper()
    elif hk.pad.is_padfn(padding):
      self.padding = hk.pad.create_from_padfn(padding=padding,
                                              kernel=self.kernel_shape,
                                              rate=self.kernel_dilation,
                                              n=self.num_spatial_dims)
    else:
      self.padding = hk.pad.create_from_tuple(padding, self.num_spatial_dims)

  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      precision: Optional[lax.Precision] = None,
  ) -> jnp.ndarray:
    """Connects ``ConvND`` layer.
    Args:
      inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
        or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
      precision: Optional :class:`jax.lax.Precision` to pass to
        :func:`jax.lax.conv_general_dilated`.
    Returns:
      An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
        unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
        and rank-N+2 if batched.
    """

    unbatched_rank = self.num_spatial_dims + 1
    allowed_ranks = [unbatched_rank, unbatched_rank + 1]
    if inputs.ndim not in allowed_ranks:
      raise ValueError(f"Input to ConvND needs to have rank in {allowed_ranks},"
                       f" but input has shape {inputs.shape}.")

    unbatched = inputs.ndim == unbatched_rank
    if unbatched:
      inputs = jnp.expand_dims(inputs, axis=0)

    if inputs.shape[self.channel_index] % self.feature_group_count != 0:
      raise ValueError(f"Inputs channels {inputs.shape[self.channel_index]} "
                       f"should be a multiple of feature_group_count "
                       f"{self.feature_group_count}")
    w_shape = self.kernel_shape + (
        inputs.shape[self.channel_index] // self.feature_group_count,
        self.output_channels)

    if self.mask is not None and self.mask.shape != w_shape:
      raise ValueError("Mask needs to have the same shape as weights. "
                       f"Shapes are: {self.mask.shape}, {w_shape}")

    w_init = self.w_init
    if w_init is None:
      fan_in_shape = np.prod(w_shape[:-1])
      stddev = 1. / np.sqrt(fan_in_shape)
      w_init = CmplxTruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

    if self.mask is not None:
      w *= self.mask

    out = lax.conv_general_dilated(inputs,
                                   w,
                                   window_strides=self.stride,
                                   padding=self.padding,
                                   lhs_dilation=self.lhs_dilation,
                                   rhs_dilation=self.kernel_dilation,
                                   dimension_numbers=self.dimension_numbers,
                                   feature_group_count=self.feature_group_count,
                                   precision=precision)

    if self.with_bias:
      if self.channel_index == -1:
        bias_shape = (self.output_channels,)
      else:
        bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
      b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    if unbatched:
      out = jnp.squeeze(out, axis=0)
    return out




class Cmplx_Conv2D(Cmplx_ConvND):
  """Two dimensional convolution."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      rate: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]], hk.pad.PadFn,
                     Sequence[hk.pad.PadFn]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NHWC",
      mask: Optional[jnp.ndarray] = None,
      feature_group_count: int = 1,
      name: Optional[str] = None,
  ):
    """Initializes the module.
    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 2.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 2. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length 2. 1 corresponds to standard ND convolution,
        ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or
        a callable or sequence of callables of length 2. Any callables must
        take a single integer argument equal to the effective kernel size and
        return a list of two integers representing the padding before and after.
        See haiku.pad.* for more details and example functions.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NHWC`` or ``NCHW``. By
        default, ``NHWC``.
      mask: Optional mask of the weights.
      feature_group_count: Optional number of groups in group convolution.
        Default value of 1 corresponds to normal dense convolution. If a higher
        value is used, convolutions are applied separately to that many groups,
        then stacked together. This reduces the number of parameters
        and possibly the compute for a given ``output_channels``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      name: The name of the module.
    """
    super().__init__(
        num_spatial_dims=2,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        feature_group_count=feature_group_count,
        name=name)


def _infer_shape(
    x: jnp.ndarray,
    size: Union[int, Sequence[int]],
    channel_axis: Optional[int] = -1,
) -> Tuple[int, ...]:
  """Infer shape for pooling window or strides."""
  if isinstance(size, int):
    if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
      raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
    if channel_axis and channel_axis < 0:
      channel_axis = x.ndim + channel_axis
    return (1,) + tuple(size if d != channel_axis else 1
                        for d in range(1, x.ndim))
  elif len(size) < x.ndim:
    # Assume additional dimensions are batch dimensions.
    return (1,) * (x.ndim - len(size)) + tuple(size)
  else:
    assert x.ndim == len(size)
    return tuple(size)


def to_dimension_numbers(
    num_spatial_dims: int,
    channels_last: bool,
    transpose: bool,
) -> lax.ConvDimensionNumbers:
  """Create a `lax.ConvDimensionNumbers` for the given inputs."""
  num_dims = num_spatial_dims + 2

  if channels_last:
    spatial_dims = tuple(range(1, num_dims - 1))
    image_dn = (0, num_dims - 1) + spatial_dims
  else:
    spatial_dims = tuple(range(2, num_dims))
    image_dn = (0, 1) + spatial_dims

  if transpose:
    kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
  else:
    kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

  return lax.ConvDimensionNumbers(lhs_spec=image_dn, rhs_spec=kernel_dn,
                                  out_spec=image_dn)

  


_VMAP_SHAPE_INFERENCE_WARNING = (
  "When running under vmap, passing an `int` (except for `1`) for "
  "`window_shape` or `strides` will result in the wrong shape being inferred "
  "because the batch dimension is not visible to Haiku. Please update your "
  "code to specify a full unbatched size. "
  ""
  "For example if you had `pool(x, window_shape=3, strides=1)` before, you "
  "should now pass `pool(x, window_shape=(3, 3, 1), strides=1)`. "
  ""
  "Haiku will assume that any additional dimensions in your input are "
  "batch dimensions, and will pad `window_shape` and `strides` accordingly "
  "making your module support both batched and per-example inputs."
)

def _warn_if_unsafe(window_shape, strides):
  unsafe = lambda size: isinstance(size, int) and size != 1
  if unsafe(window_shape) or unsafe(strides):
    warnings.warn(_VMAP_SHAPE_INFERENCE_WARNING, DeprecationWarning)


    
def cmplx_max_pool(
    value: jnp.ndarray,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    channel_axis: Optional[int] = -1,
) -> jnp.ndarray:
  """Max pool.
  Args:
    value: Value to pool.
    window_shape: Shape of the pooling window, an int or same rank as value.
    strides: Strides of the pooling window, an int or same rank as value.
    padding: Padding algorithm. Either ``VALID`` or ``SAME``.
    channel_axis: Axis of the spatial channels for which pooling is skipped,
      used to infer ``window_shape`` or ``strides`` if they are an integer.
  Returns:
    Pooled result. Same rank as value.
  """
  if padding not in ("SAME", "VALID"):
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  _warn_if_unsafe(window_shape, strides)
  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  cmplx_max = lax.max#lambda x, y: x if lax.abs(x) >= lax.abs(y) else y

  return lax.reduce_window(value, -jnp.inf, cmplx_max, window_shape, strides,
                           padding)




################ POOLING LAYERS #######################################################


class MaxMagnitude_Pooling(hk.Module):
  """Max pool.
  Equivalent to partial application of :func:`max_pool`.
  """

  def __init__(
      self,
      window_shape: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]],
      padding: str,
      channel_axis: Optional[int] = -1,
      name: Optional[str] = None,
  ):
    """Max pool.
    Args:
    window_shape: Shape of window to pool over. Same rank as value or ``int``.
    strides: Strides for the window. Same rank as value or ``int``.
    padding: Padding algorithm. Either ``VALID`` or ``SAME``.
    channel_axis: Axis of the spatial channels for which pooling is skipped.
    name: String name for the module.
    """
    super().__init__(name=name)
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.channel_axis = channel_axis
    
  def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
    return cmplx_max_pool(value, self.window_shape, self.strides,
                          self.padding, self.channel_axis)





  

################## NORMALIZATION LAYERS ####################################################


class Cmplx_Normalization(hk.Module):
  """Basic implementation of a Complex normalization layer, that normalize all the input data
  to a unitary magnitude, leaving the phase untouched."""

  def __init__(
      self,
      name: Optional[str] = None,
  ):
    """Constructs the Cmplx_Normalization module.
    Args:
      name: Name of the module.
    """
    super().__init__(name=name)
    
  def __call__(
      self,
      x: jnp.ndarray,
  ) -> jnp.ndarray:
    """Implementation of a complex normalization."""
    norm = jnp.absolute( x.reshape(x.shape[0],-1) ).mean(axis=1).reshape(-1,1)

    return x / norm
    

  
class CmplxBatchNorm(hk.Module):

    def __init__(
        self,
        create_scale: bool,
        create_offset: bool,
        decay_rate: float,
        eps: float = 1e-5,
        scale_rr_init: float = 1./jnp.sqrt(2.),
        scale_ri_init: float = 0.,
        scale_ii_init: float = 1./jnp.sqrt(2.),
        offset_init: Optional[hk.initializers.Initializer] = None,
        axis: Optional[Sequence[int]] = None,
        data_format: str = "channels_last",
        name: Optional[str] = None,
    ):

        super().__init__(name=name)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`")
                 
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_rr_init = hk.initializers.Constant(scale_rr_init)
        self.scale_ri_init = hk.initializers.Constant(scale_ri_init)
        self.scale_ii_init = hk.initializers.Constant(scale_ii_init)
        self.offset_init = offset_init or jnp.zeros
        self.axis = axis
        self.channel_index = get_channel_index(data_format)
        self.mean_ema = hk.ExponentialMovingAverage(decay_rate, name="mean_ema")
        self.cov_ema = hk.ExponentialMovingAverage(decay_rate, name="cov_ema")


    def __call__(
        self,
        inputs: jnp.ndarray,
        is_training: bool,
        test_local_stats: bool = False,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """The implementation is based on an equivalent module written for Pytorch
        https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py."""

        if self.create_scale and scale is not None:
            raise ValueError(
                "Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`.")

        channel_index = self.channel_index
        channels_first = False
        if channel_index == 1: channels_first = True
        if channel_index < 0:
            channel_index += inputs.ndim

        if self.axis is not None:
            axis = self.axis
        else:
            axis = [i for i in range(inputs.ndim) if i != channel_index]

        # Split input into its real and imaginary components, now it has shape [2,..]
        inputs = jnp.array([inputs.real, inputs.imag])

        # Shift the axis to the new shape
        channel_index += 1
        axis = [i+1 for i in axis]


        if is_training or test_local_stats:

            # Compute the mean for each channel and for real/imag
            mean = jnp.mean(inputs, axis, keepdims=True)           # shape = (2,1,C,1,..,1) or (2,1,..,1,C)

            # Center the inputs
            centered_inputs = inputs - mean

            # Compute the variances (Add small epsilon to increase stability)
            variances = (centered_inputs * centered_inputs).mean(axis) + self.eps  # shape = (2,C)
            Var_Rez, Var_Imz = variances[0], variances[1]   # shape = (C,)

            # Compute the covariances
            Cov_ReIm = Cov_ImRe = (centered_inputs[0] * centered_inputs[1]).mean([a-1 for a in axis])  # shape = (C,)

            # Construct the covariance matrix
            covariance_matrix = jnp.array( [[Var_Rez, Cov_ReIm], [Cov_ImRe, Var_Imz]] ).reshape(2,2,-1)   # shape = (2,2,C)

        else:
            mean = self.mean_ema.average

            # Center the inputs
            centered_inputs = inputs - mean
            
            covariance_matrix = self.cov_ema.average
            Var_Rez, Cov_ReIm, Cov_ImRe, Var_Imz = covariance_matrix.reshape(4,-1)

        # Update the moving averages
        if is_training:
            self.mean_ema(mean)
            self.cov_ema(covariance_matrix)


        # To construct the inverse square root of the covariance matrix, without explicit inversion, we follow
        # the instructions at https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        sqrt_det = jnp.sqrt(Var_Rez * Var_Imz - Cov_ReIm * Cov_ImRe)
        sqrt_tr  = jnp.sqrt(Var_Rez + Var_Imz + 2*sqrt_det)
        denom = sqrt_det * sqrt_tr

        inverse_root_covmat = jnp.array([[Var_Imz + sqrt_det, - Cov_ReIm],
                                         [- Cov_ImRe, Var_Rez + sqrt_det]]).reshape(2,2,-1)   # shape = (2,2,C)

        inverse_root_covmat /= denom        

        
        # Normalize the input data
        if channels_first:
            einstein_formula = 'ijk,jlk...->ilk...'
        else:
            einstein_formula = 'ij...,j...->i...'
            
        normalized_input = jnp.einsum(einstein_formula, inverse_root_covmat, centered_inputs)

        
        w_shape = Var_Rez.shape
        w_dtype = Var_Rez.dtype
        b_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]


        if self.create_scale:
            scale_rr = hk.get_parameter("scale_rr", w_shape, w_dtype, self.scale_rr_init)
            scale_ri = hk.get_parameter("scale_ri", w_shape, w_dtype, self.scale_ri_init)
            scale_ii = hk.get_parameter("scale_ii", w_shape, w_dtype, self.scale_ii_init)
            scale = jnp.array([[scale_rr, scale_ri],[scale_ri, scale_ii]], dtype=w_dtype).reshape(2,2,-1)
        elif scale is None:
            scale = np.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter("offset", b_shape, w_dtype, self.offset_init)
        elif offset is None:
            offset = np.zeros([], dtype=w_dtype)


        # Apply parameters transform
        output = jnp.einsum(einstein_formula, scale, normalized_input) + offset
        
        # Reconstruct the complex normalized input
        return jax.lax.complex(output[0], output[1])
