## Complex-Valued Deep Learning for Condition Monitoring

Brief introduction.



### Main Results achieved

* As recently found by [Barrachina et al.](https://arxiv.org/abs/2009.08340v2), complex-valued neural networks seems to provide better classification accuracies, with respect to equivalent real valued models, when trying to distinguish among two complex distributions that differ only by internal correlations among their real and imaginary parts. This correlation is quantified by a complex variable known as circularity coefficient.
<img src="circularity_results.png" alt="Circularity Classification Performances" title="Circularity Classification Performances" align="center" width=1000 />

* The domain adaptation algorithm known as [`Wasserstein Distance Guided Representation Learning (WDGRL)`](https://arxiv.org/abs/1707.01217) can be extended, with good performances, also in the complex domain. That's because the metric used (the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric)) can be extended also in C.

### JAX

One of the main obstacles in the development and diffusion of complex-valued deep learning is the lack of support provided by existing libraries:
* the most popular hardware acceleration architectures, CUDA and CuDNN doesn’t own a native support for complex-valued data types;
* moder libraries like Tensorflow and Pytorch does not fully support many operations needed (at least for this work) even if they are starting implementing them.

For this reasons, we relied on a more recent library, [`JAX`](https://github.com/google/jax), developed by Google DeepMind. JAX is Autograd and XLA (a domain-specific compiler for linear algebra designed for TensorFlow models), brought together for high-performance numerical computing and machine learning research:
* it supports complex differentiation for both holomorphic and non-holomorphic functions;
* it is extremely optimized, with XLA + JIT that partially compensate the lack of native hardware acceleration;
* many complex operations/layers are already supported and well defined.

More specifically, we have written the library over [`Haiku`](https://github.com/deepmind/dm-haiku), another library built on top of JAX with the purpose of covering the same role that Sonnet (widely used ad DeepMind) has for Tensorflow, and to simplify the approach of users that are familiar with object oriented programming.

At the author's knowledge, at least at the beginning of the work, Haiku was the most ahead library in this sense. However, this is a research area that mutate quite fast, and it is better to remain aware of recent updates. For example, we believe that there are other similar alternative to Haiku like [`Flax`](https://github.com/google/flax), that can work as well.


### Repository Content

- [`thesis_censored`](thesis_censored.pdf): 
- [`papers`](papers):
- [`Notebook_Analysis`](Notebook_Analysis):
- [`complex_nn`](complex_nn):


### Abstract

At present, the vast majority of deep learning architectures are based on real-valued operations and representations. However, recent works and fundamental theoretical analyses suggest that complex numbers can have richer expressiveness: many deterministic wave signals, such as seismic, electrical, or vibrational, contain information in their phase, which risks being lost when studied using a real-valued model. However, despite their attractive properties and potential, only recently complex-valued algorithms have started to be introduced in the deep neural networks frameworks.
In this work, we move forward in this direction developing and implementing a coherent and working structure to train complex-valued models, remaining rigorous from a mathematical perspective but, at the same time, seeking for stability and accuracy of the training process.
As a first application of this solution, we show the results obtained applying complex-valued deep neural networks for condition monitoring in industrial applications. Different Deep Network architectures have been trained on vibrational signals extracted from sensors attached to gearmotors to detect failures. Finally, we compare the performances obtained with real and complex-valued neural networks.

### Thesis organization

The thesis is organized into two big parts: part I is dedicated to the theoretical analysis and implementation of a complex-valued deep learning framework, while part II proceed implementing, in details, such methodology to a real-world problem, i.e. condition monitoring in industrial applications. 
In particular:
* in chapter 2 there is a brief theoretical introduction of complex analysis, with particular attention given to the notions of complex differentiability and circularity; also, the theoretical advantages that complex-valued deep learning should provide over the real counterpart are introduced.
* in chapter 3, the project continues with a more precise formulation of the framework that is trying to build, first examining a working complex backpropagation algorithm, and then with some possible extents, in the complex domain, of the most common machine learning layers and activations.
* in chapter 4 a practical implementation of the operations defined is provided, together with the comparison with an equivalent real-valued procedure; there is also an interesting section examining the impact of the circularity quotient when training complex models.
* in chapter 5, the framework developed will be tested over a real world situation, i.e. the problem of condition monitoring in industrial applications; in particular, it will work over datasets of vibration signals, provided by Bonfiglioli, an important gearmotors producer, comparing the effective performances achieved with both real and complex-valued models.
* in chapter 6, finally, it will be proposed an of a known domain adaptation algorithm that seems to be easily extendable also in the complex domain.

### Important Clarification

Unfortunately, Bonfiglioli S.P.A. didn't allow us to make public the results obtained using their data, and so we had to censor a few important sections in this thesis. For those parts, you can only trust our words.


### Acknowledgement

This work has been inspired by the PHD thesis of Patrick Virtue "[Complex-valued Deep Learning with Applications to Magnetic Resonance Image Synthesis](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-130.html)", from which we took some fundamental ideas behind the framework development, and that we expanded into different directions. 
Furthermore, we provide a more modern implementation of complex-valued layers, activations, optimizers and initializers, based on Python, and not on Caffè.
