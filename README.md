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
- Install this caffe
- Examples for CIFAR-100 in ./examples/cifar100
```
  ./train.sh 0 simple  # 0 is the GPU id, simple is the folder containing network definitions and solver
```

## Layer details
- Specify margin parameter &alpha; and likelihood weight &lambda;, which is *margin_mul* and *center_coef* in the layer param, respectively.  
```
    margin_mul {
      policy: STEPUP
      value: 0.1
      step: 5000 
      gamma: 2
      max: 0.3 
}
```
This specifies a gradually growing value for &alpha; (multiplied by 2 every 5000 iterations, with initial value 0.1 and final maximum value 0.3), which is helpful for training.

- other indicators
```
update_sigma: false
```
Fix the variances to initial values (1.0).

```
isotropic: true
```
The variances of different dimensions are identical.
  
  
More contents under construction ......

## Data
We've described how the data is pre-processed in our paper. For example, the CIFAR-100 training data (32x32) is padded to 40x40 with zero pixels and then randomly cropped with a 32x32 window for training.  
In the CIFAR-100 example, we use data in HDF5 format. You can choose other formats, changing the data layer accordingly.

The CIFAR100 training data (with or without augmentation) and test data are available for download in [Baidu Drive](https://pan.baidu.com/s/1tvB8UVlrgKvzVYJHYXc3Aw).

## Citations
If you find this work useful, please consider citing it.
```
@inproceedings{LGM2018,
  title={Rethinking Feature Distribution for Loss Functions in Image Classification},
  author={Wan, Weitao and Zhong, Yuanyi and Li, Tianpeng and Chen, Jiansheng},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```
