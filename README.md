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
    <img src="https://github.com/Alonso94/GSoC-DGP/blob/master/GSoC1.png" width="700"><br>
    - [Gaussian Process Latent Variable Model](https://github.com/Alonso94/GSoC-DGP/blob/master/GP_LVM.ipynb)<br>
   Fit a Gaussian process to index points in a latent space (latent index points), instead of fitting it directly to the data. After training the model we can us Gaussian Process Regression model to sample from the latent space and get new observations.<br>
    Latent space before training and after:<br>
      <img src="https://github.com/Alonso94/GSoC-DGP/blob/master/GSoC2.png" width="250">
      <img src="https://github.com/Alonso94/GSoC-DGP/blob/master/GSoC3.png" width="250"><br>
3. Read thoroughly the Vanilla Deep Gaussian Process paper [Damianou & Lawrence, 2013](http://proceedings.mlr.press/v31/damianou13a.pdf), and Doubly Stochastic Variational Inference for Deep Gaussian Processes [Salimbeni & Deisenroth, 2017](https://arxiv.org/abs/1705.08933), review the needed literature and discuss the ideas with the mentor.
4. Start to code the Deep Gaussian Processes in Tensorflow Probability; a lot of time spent on rewrite the code again and again, and on debugging and changing some parts of it; milestones from the process of coding;
    - [Single layer exact GP](https://github.com/Alonso94/GSoC-DGP/blob/master/Single_layer_GP.ipynb):<br>
    It was just a try to make a layer class using the aforementioned Gaussian process regression.
    - [Deep Gaussian Process class - 2nd evaluation](https://github.com/Alonso94/GSoC-DGP/blob/master/First_DGP_2nd_evaluation.ipynb):<br>
    The first try to make DGP class that stack substances of a layer class.
    - [Second edition of DGP class - early August](https://github.com/Alonso94/GSoC-DGP/blob/master/second_DGP_early_August.ipynb):<br>
    This version comes after working on the last one, trying to make LVM layers, and train each layer seperately, it was a loss of time because I had done something wrong in it.
    - [Third edition of DGP class - final evaluation](https://github.com/Alonso94/GSoC-DGP/blob/master/DGP_final_evaluation_closest_to_work.ipynb):<br>
    I have corrected the loss computation and the training idea, trying with the prediction method but still have some problems in it.<br>
    The training process is woking now, a sample plot of the loss during training:<br>
    <img src="https://github.com/Alonso94/GSoC-DGP/blob/master/GSoC4.png" width="500">
