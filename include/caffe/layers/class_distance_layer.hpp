#ifndef CAFFE_CLASS_DISTANCE_LAYER_HPP_
#define CAFFE_CLASS_DISTANCE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ClassDistanceLayer : public Layer<Dtype> {
 public:
  explicit ClassDistanceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ClassDistance"; }
  //virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> dist_;
  Dtype eps_; // to prevent division by 0
  //bool transpose_;  ///< if true, assume transposed weights
  //Blob<Dtype> tmp_;
  //Dtype margin_mul_, margin_add_;
  //int iterations_;

  Param<Dtype> m_mul_, m_add_;

  /*enum {
      L2 = 0,  // euclidean L2 distance ^2
      IP = 1,  // inner product
      L1 = 2,  // L1 distance
  } metric_;*/
  //ClassDistanceParameter_Metric metric_;
  //Dtype center_coef_;

  //Blob<unsigned int> rand_vec_;
};

}  // namespace caffe

#endif  // CAFFE_CLASS_DISTANCE_LAYER_HPP_
