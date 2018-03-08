# L-GM-loss
Implementation of our CVPR 2018 paper "Rethinking Feature Distribution for Loss Functions in Image Classification"

We implemented it in Caffe. I also have a tensorflow implementation but it hasn't been tested yet.
Now I'm rearranging the code to make it look neat and (hopefully) a bit more beautiful.

To appear soon.

### Abstract

We propose a large-margin Gaussian Mixture (L-GM) loss for deep neural networks in classification tasks.
Different from the softmax cross-entropy loss, our proposal is established on the assumption that the deep features of the training set follow a Gaussian Mixture distribution.
By involving a classification margin and a likelihood regularization, the L-GM loss facilitates both a high classification performance and an accurate modeling of the training feature distribution.
As such, the L-GM loss is superior to the softmax loss and its major variants in the sense that besides classification, it can be readily used to distinguish abnormal inputs, such as the adversarial examples, based on their features' likelihood to the training feature distribution.
Extensive experiments on various recognition benchmarks like MNIST, CIFAR, ImageNet and LFW, as well as on adversarial examples demonstrate the effectiveness of our proposal.
