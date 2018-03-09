#include <vector>

#include "caffe/layers/new_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NewLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "Inputs must have the same dimension.";
  //diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NewLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
    const Dtype* predict = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const int M = bottom[0]->count(1);
    const Dtype th = Dtype(1) / M;

    Dtype loss = 0;
    for (int i = 0; i < bottom[0]->num(); ++i) {
        Dtype d = 1 - predict[i*M + int(label[i])];
        
        //loss += std::pow(d, 1.5);

        loss += d * d;

        //loss += d * d * d;

        //loss += -d;
        
        /*Dtype d = predict[i*M + int(label[i])];
        if (d < th)
            loss += 2 - 2 * d;
        else
            loss += 1 - d;*/

        //Dtype d = predict[i*M + int(label[i])];
        //loss -= log(d);
    }
    top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
void NewLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
    const Dtype* predict = bottom[0]->cpu_data();
    Dtype* predict_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    const int M = bottom[0]->count(1);
    const Dtype th = Dtype(1) / M;

    const Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
    //const Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num() * 2;
    
    caffe_set(bottom[0]->count(), Dtype(0), predict_diff);
    //printf("(%d)", M);
    for (int i = 0; i < bottom[0]->num(); ++i) {
        int ii = i*M + int(label[i]);
        //printf("[%d]%.1f ", i,predict[ii]);
        Dtype d = 1 - predict[ii];

        //predict_diff[ii] = scale * (-Dtype(1.5)*std::pow(d,0.5));

        predict_diff[ii] = scale * (-2*d);

        //predict_diff[ii] = scale * (-3*d*d);

        //predict_diff[ii] = scale * (-1);

        /*Dtype d = predict[i*M + int(label[i])];
        if (d < th)
            predict_diff[ii] = scale * (-2);
        else
            predict_diff[ii] = scale * (-1);*/
        
        //Dtype d = predict[ii];
        //predict_diff[ii] = -std::min(Dtype(1),1/d * loss);
        //predict_diff[ii] = -scale / d;
    }
}

#ifdef CPU_ONLY
STUB_GPU(NewLossLayer);
#endif

INSTANTIATE_CLASS(NewLossLayer);
REGISTER_LAYER_CLASS(NewLoss);

}  // namespace caffe
