#ifndef CAFFE_MARGIN_SOFTMAX_LOSS_LAYER_HPP_
#define CAFFE_MARGIN_SOFTMAX_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
//#include "caffe/layers/softmax_layer.hpp"

#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
class MarginSoftmaxLossLayer : public LossLayer<Dtype> {
 public:

  explicit MarginSoftmaxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MarginSoftmaxLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  //virtual inline int MinTopBlobs() const { return 1; }
  //virtual inline int MaxTopBlobs() const { return 2; }

  /*void print() {
      const Dtype *q = prob_.cpu_data();
      for (int i = 0; i < 2 * 3; ++i) {
          printf("%f ", q[i]);
      }
      printf(";  ");
      q = prob_.cpu_diff();
      for (int i = 0; i < 2 + 2 + 2 + 3 + 3; ++i) {
          printf("%f ", q[i]);
      }
      printf("\n");
  }*/
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int N, F, P;  // N_samples, Feature length, Predictions
  //Dtype lambda;
  Param<Dtype> lambda_;
  Blob<Dtype> ip_, prob_;

  //const int m_ = 4;
};

}  // namespace caffe

#endif  // CAFFE_MARGIN_SOFTMAX_LOSS_LAYER_HPP_
