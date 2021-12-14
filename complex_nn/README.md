## Complex_nn

This small Python library, written on top of `JAX` and `dm-haiku`, has been realized in order to provide a fast implementation and run of a complex-valued neural network for classification tasks. It implements the main complex layers, activations and initializers discussed in the thesis project, together with high-level wrappers of more composite deep learning structures to train.

Library:
  * [`activations.py`](activations.py) -> collection of complex-valued activation functions;
  * [`initializers.py`](initializers.py) -> reformulation of some dm-haiku initializers to handle complex-valued initialization of network's weights;
  * [`layers.py`](layers.py) -> implementation of the fundamental complex-valued layers discussed during the thesis work;
  * [`optimizers.py`](optimizers.py) -> implementation of a complex-valued version of the Adam optimizer;
  * [`metrics.py`](metrics.py) -> definitions of a few metrics for complex-valued data types;
  * [`haiku_ml_utils.py`](haiku_ml_utils.py) -> utility functions for complex-valued models built over Haiku;
  * [`utils.py`](utils.py) -> a few utility functions.

Wrappers:
  * [`Haiku_Classifier`](Classifier_wrapper.py) -> an high-level structure to handle and train classification tasks.
  * [`WDGRL`](wdgrl.py) -> an high-level structure to setup and run a Wasserstein Distance Guided Representation Learning algorithm for complex data.
