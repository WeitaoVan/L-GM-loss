#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/class_distance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClassDistanceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const ClassDistanceParameter& param = this->layer_param_.class_distance_param();

  N_ = param.num_output();
  bias_term_ = param.bias_term();

  K_ = bottom[0]->count(1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    // Initialize the weights
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;

    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // Initialize the sigmas
    if (param.isotropic()){
      vector<int> sigma_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(sigma_shape));
    }
    else {
      vector<int> sigma_shape(2);
      sigma_shape[0] = N_;
      sigma_shape[1] = K_;
      this->blobs_[1].reset(new Blob<Dtype>(sigma_shape));
    }
    
    // fill the sigmas. We use the bias_filler.
    shared_ptr<Filler<Dtype> > sigma_filler(GetFiller<Dtype>(param.bias_filler()));
    sigma_filler->Fill(this->blobs_[1].get());
    // epsilon
    eps_ = 0.0001;

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);


  this->m_mul_.init(param.margin_mul());

  LOG(INFO) << "metric: " << param.metric() << ", alpha: " << m_mul_ 
      << ", likelihood_coef = " << param.center_coef();
}

template <typename Dtype>
void ClassDistanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
    const int axis = 1;// bottom[0]->CanonicalAxisIndex(
//      this->layer_param_.class_distance_param().axis());
  const int new_K = bottom[0]->count(1);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();

  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  dist_.Reshape(top_shape);

}

template <typename Dtype>
void ClassDistanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    LOG(FATAL) << "not implemented";

}

template <typename Dtype>
void ClassDistanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    LOG(FATAL) << "not implemented";
}

#ifdef CPU_ONLY
STUB_GPU(ClassDistanceLayer);
#endif

INSTANTIATE_CLASS(ClassDistanceLayer);
REGISTER_LAYER_CLASS(ClassDistance);

}  // namespace caffe
