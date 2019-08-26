# Google Summer of Code 2019
### Project : Deep Gaussian Process in Tensorflow Probability
### Orginization : TensorFlow
### Mentor : Christopher Sutter

I have worked this summer on implemnting Deep Gaussian Process in TensorFlow probability. The done work depends extensively on the [Double Stochastic Variational Inference for Deep Gaussian Processes](https://arxiv.org/abs/1705.08933) paper.

## Acheived work:
1. Review the needed parts of TensorFlow Probability;<br> 
  - [Base Distribution class](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/distribution.py)
  - [Gaussian Process distribution](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/gaussian_process.py)
  - [Gaussian Process Regression Model](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/gaussian_process_regression_model.py)
  - [Variational Gaussian Porcess distribution](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/variational_gaussian_process.py)
2. Work on demonstratoins for the last parts; preparing colab's notebooks with comments explaining the work:<br>
  - [Gaussian process regression for toy data](https://github.com/Alonso94/GSoC-DGP/blob/master/Exact_gpr.ipynb):<br>
  Using Gaussian Process prior and train it to fit a disturbed sin wave, after that use Gaussian Process Regression model with the trained kernel to draw samples; result is in the following image<br>
  <img src="https://github.com/Alonso94/GSoC-DGP/blob/master/GSoC1.png" width="700">
  - [Gaussian Process Latent Varialbe Model](https://github.com/Alonso94/GSoC-DGP/blob/master/GP_LVM.ipynb):<br>
  Fit a Gaussian process to index points in a latent space (latent index points), instead of fitting it directly to the data. After training the model we can us Gaussian Process Regression model to sample from the latent space and get new observations.<br>
  Latent space before training and after:<br>
    <img src="https://github.com/Alonso94/GSoC-DGP/blob/master/GSoC2.png" width="300">
    <img src="https://github.com/Alonso94/GSoC-DGP/blob/master/GSoC3.png" width="300">
3. 
