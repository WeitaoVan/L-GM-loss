#include <algorithm>
#include <vector>

#include "caffe/layers/margin_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MarginContrastiveLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  //CHECK_EQ(bottom[1]->height(), 1);
  //CHECK_EQ(bottom[1]->width(), 1);
  //CHECK_EQ(bottom[2]->channels(), 1);
  //CHECK_EQ(bottom[2]->height(), 1);
  //CHECK_EQ(bottom[2]->width(), 1);
  
  M_ = bottom[0]->num();
  N_ = bottom[0]->channels();
  margin_ = this->layer_param_.contrastive_loss_param().margin();
}

template <typename Dtype>
void MarginContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MarginContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MarginContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(MarginContrastiveLossLayer);
REGISTER_LAYER_CLASS(MarginContrastiveLoss);

}  // namespace caffe
