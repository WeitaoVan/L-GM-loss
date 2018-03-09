#include <algorithm>
#include <vector>

#include "caffe/layers/gather_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GatherLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void GatherLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const GatherParameter& param = this->layer_param_.gather_param();
  const bool inverse = param.inverse();
  if (!inverse){
    vector<int> top_shape = bottom[1]->shape();
    top[0]->Reshape(top_shape);
  }
  else{
    vector<int> top_shape = bottom[0]->shape();
    top_shape[1] -= 1;
    top[0]->Reshape(top_shape);
  }
}

template <typename Dtype>
void GatherLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void GatherLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(GatherLayer);
#endif

INSTANTIATE_CLASS(GatherLayer);
REGISTER_LAYER_CLASS(Gather);

}  // namespace caffe
