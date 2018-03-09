# L-GM-loss
Implementation of our CVPR 2018 paper "[Rethinking Feature Distribution for Loss Functions in Image Classification](https://arxiv.org/abs/1803.02988)".  
Paper authors: Weitao Wan, Yuanyi Zhong, Tianpeng Li, Jiansheng Chen.

We implemented it in Caffe. I also have a tensorflow implementation but it hasn't been fully tested yet.
Now I'm rearranging the code to make it look neat and (hopefully) a bit more beautiful.


Code is written by Yuanyi Zhong and Weitao Wan.


## Abstract

We propose a large-margin Gaussian Mixture (L-GM) loss for deep neural networks in classification tasks.
Different from the softmax cross-entropy loss, our proposal is established on the assumption that the deep features of the training set follow a Gaussian Mixture distribution.
By involving a classification margin and a likelihood regularization, the L-GM loss facilitates both a high classification performance and an accurate modeling of the training feature distribution.
As such, the L-GM loss is superior to the softmax loss and its major variants in the sense that besides classification, it can be readily used to distinguish abnormal inputs, such as the adversarial examples, based on their features' likelihood to the training feature distribution.
Extensive experiments on various recognition benchmarks like MNIST, CIFAR, ImageNet and LFW, as well as on adversarial examples demonstrate the effectiveness of our proposal.

<img src="https://github.com/WeitaoVan/L-GM-loss/blob/master/distribution.png" width="800">

## Instructions
- Install this caffe according to caffe's instructions
- Examples for CIFAR-100 in ./examples/cifar100
```
  ./train.sh 0 simple  # 0 is the GPU id, simple is the folder containing network definitions and solver
```
