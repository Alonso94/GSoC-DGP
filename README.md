# Google Summer of Code 2019
### Project : Deep Gaussian Process in Tensorflow Probability
### Orginization : TensorFlow
### Mentor : Christopher Sutter

I have worked this summer on implemnting Deep Gaussian Process in TensorFlow probability. The done work depends extensively on the [Doubly Stochastic Variational Inference for Deep Gaussian Processes](https://arxiv.org/abs/1705.08933) paper.

## Deep Gaussian Process in TFP:
A __Gaussian process (GP)__ is an indexed collection of random variables, any finite collection of which are jointly Gaussian.<br>
Gaussian processes are a good choice for function approximation as they are flexible, robust to overfitting and provide well-calibrated predictive uncertainty.GPs have been used in machine learning and it could improve the efficiency of learning process (GPs are used with small datasets).<br>
A __Deep Gaussian Process (DGP)__ is a multi-layer generalization of GPs, the main advavntage of stacking layers in DGPs is adding flexibility to the probabilistic model. In other words, we don't have to care so much about the type of the kernel and its parameters, similar to the advatage of deep neural network over the shallow one.<br>
As shown in the most of recent papers realted to DGPs, DGPs outperform GPs in performance (or at least have similar performance), also DGP can tackle some problems of GP when working with large datasets, and with disontinuouty.<br>

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
    

## DGP code:
### [DGP_final_evaluation_closest_to_work.ipynb](https://github.com/Alonso94/GSoC-DGP/blob/master/DGP_final_evaluation_closest_to_work.ipynb)
To implement DGP in Tensorflow probability, we have made two classes:
1. __Layer class__:
    each layer can be seen as a an ordinaru GP (or set of GPs), we have used __Variational Gaussian Processes (VGPs)__, in our implementation, as they provide better efficiency in computation with a plausible approximation.<br>
    This class has the following methods:<br>
    - __Initialization method__: init() function <br>
    We define the parameters of the kernel (amplitude and lengthscale), the variational locations and scale of the observations for each GP. Also each GP node in our DGP has a prior (ordinary GP distribution with zero mean and identity covariance) and a variational posterior (VGP), takes its input from an input placeholder, uses inducing index points from that index points.<br>
    - __Sampling method__ : sample() function <br>
    draw samples from the posterior(s) and stack them in case of multioutput GP.
    - __Updating the prior methid__: update_prior() method<br>
    After every training iterations, we have to update the prior to be the posterior, to compare it afterwards with the new posterior (needed to compute the loss in KL divergence term).
The layer class can be seen as the following:
<img src="https://github.com/Alonso94/GSoC-DGP/blob/master/DGP_layer.png" width="400">
