#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Spatial Transformer Network (Max Jaderberg etc, Spatial Transformer Networks.)
 * current version: affine transform + bilinear sampling
 */
template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype> {
 public:
  explicit SpatialTransformerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "SpatialTransformer"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; };
  virtual inline int MinTopBlobs() const { return 1; }
 
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // cpu only <<
  Blob<Dtype> target_; // 1*3*(H*W), different channels and nums have the same coordinate system
  // >>

  Blob<Dtype> source_; // N*(2*H*W), channels are shared
  int num_;
  int channels_;
  int height_;
  int width_;
  int output_H_, output_W_;
  int map_size_;
  int t_type_;
  
  // gpu only <<
  Blob<Dtype> theta_diff_cache_; // N*6*H*W to allow separately handling of gradients for different pixels
  Blob<Dtype> theta_diff_op_;    // H*W to sum over all pixels
  // >>
};

}  // namespace caffe

#endif  // CAFFE_CUSTOM_LAYERS_HPP_
