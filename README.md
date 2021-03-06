## Complex-Valued Deep Learning for Condition Monitoring

Nowadays, we still lack of a complete and working complex-valued deep learning framework. Even tho it has be proven to work in several areas, mainly because of mathematical and computational reasons it is still poorly used by researchers. In view of my final dissertation for the Master Degree in Physics of Data at the University of Padua, I'm proposing a possible extent of existing machine learning structures to the complex domain, based on many previous works that I'm reorganizing and reformulating in a more rigorous way, addressing and analyzing also the main problems encountered during this formulation. Furthermore, I will stress such framework over a real world problem like Condition Monitoring in Industrial application, effectively proving its efficiency and the advantages brought over equivalent real-valued approaches. In the end, we will examine an area poorly considered by researchers in the years, that is transfer learning (in particular domain adaptation) for complex-vlaued models.


### Main Results achieved

* As recently found by [Barrachina et al.](https://arxiv.org/abs/2009.08340v2), complex-valued neural networks seems to provide better classification accuracies, with respect to equivalent real valued models, when trying to distinguish among two complex distributions that differ only by internal correlations among their real and imaginary parts. This correlation is quantified by a complex variable known as circularity coefficient.
<img src="circularity_results.png" alt="Circularity Classification Performances" title="Circularity Classification Performances" align="center" width=1000 />

* Confirmed by many other works in this sense, complex-valued neural networks seems to be more robust to overfitting, at least with respect to equivalent real-valued models, especially in "bad conditions" (small training set, bad regularization, few parameters).

* The domain adaptation algorithm known as [`Wasserstein Distance Guided Representation Learning (WDGRL)`](https://arxiv.org/abs/1707.01217) can be extended, with good performances, also in the complex domain. That's because the metric used (the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric)) can be extended also in C.

### JAX

One of the main obstacles in the development and diffusion of complex-valued deep learning is the lack of support provided by existing libraries:
* the most popular hardware acceleration architectures, CUDA and CuDNN doesn???t own a native support for complex-valued data types;
* moder libraries like Tensorflow and Pytorch does not fully support many operations needed (at least for this work) even if they are starting implementing them.

For this reasons, we relied on a more recent library, [`JAX`](https://github.com/google/jax), developed by Google DeepMind. JAX is Autograd and XLA (a domain-specific compiler for linear algebra designed for TensorFlow models), brought together for high-performance numerical computing and machine learning research:
* it supports complex differentiation for both holomorphic and non-holomorphic functions;
* it is extremely optimized, with XLA + JIT that partially compensate the lack of native hardware acceleration;
* many complex operations/layers are already supported and well defined.

More specifically, we have written the library over [`Haiku`](https://github.com/deepmind/dm-haiku), another library built on top of JAX with the purpose of covering the same role that Sonnet (widely used ad DeepMind) has for Tensorflow, and to simplify the approach of users that are familiar with object oriented programming.

At the author's knowledge, at least at the beginning of the work, Haiku was the most ahead library in this sense. However, this is a research area that mutate quite fast, and it is better to remain aware of recent updates. For example, we believe that there are other similar alternative to Haiku like [`Flax`](https://github.com/google/flax), that can work as well.


### Repository Content

- [`thesis_censored`](thesis_censored.pdf): Thesis work (with some censored parts, see [Important Clarification](#Important-Clarification))
- [`Notebook_Analysis`](Notebooks_Analysis): Some Jupyter Notebooks realized to perform, in an ordered way, our analysis.
    * [`Complex_gradient_descent`](Notebooks_Analysis/Complex_gradient_descent.ipynb) -> complex gradient descent algorithm implemented in JAX and check of the steepest direction;
    * [`Complex_Valued_Deep_Learning`](Notebooks_Analysis/Complex_Valued_Deep_Learning.ipynb) -> complex-valued classification problem of the PhaseMNIST dataset;
    * [`Complex_Valued_Activation_Functions`](Notebooks_Analysis/Complex_Valued_Activation_Functions.ipynb) -> implementation and analysis of several complex-valued activation functions;
    * [`Circularity_Measures`](Notebooks_Analysis/Circularity_Measures.ipynb) -> studies on the impact of the circularity property in complex-valued classification problems;
    * [`Bearing_vibration_data_classification`](Notebooks_Analysis/Bearing_vibration_data_classification.ipynb) -> complex-valued classification problem over the Mendelay dataset (vibration signals collected in non-stationary conditions).
- [`complex_nn`](complex_nn): our own library (based on JAX) for high-level complex-valued deep learning.


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

Unfortunately, some of the results and datasets we have used are covered by confidentiality restrictions related to industrial projects, and so they had to be omitted. For this reasons, some important sections in this thesis have been censored: for those parts, you can only trust our words.


### Acknowledgement

This work has been inspired by the PHD thesis of Patrick Virtue "[Complex-valued Deep Learning with Applications to Magnetic Resonance Image Synthesis](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-130.html)", from which we took some fundamental ideas behind the framework development, and that we expanded into different directions. 
Furthermore, we provide a more modern implementation of complex-valued layers, activations, optimizers and initializers, based on Python, and not on Caff??.
