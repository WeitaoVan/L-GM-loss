//#include <algorithm>
#include <cfloat>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/margin_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#define M_PI       3.14159265358979323846

namespace caffe {

template <typename Dtype>
void MarginSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MarginSoftmaxLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    if (this->blobs_.size() == 0) {
        LossLayer<Dtype>::Reshape(bottom, top);
        N = bottom[0]->shape(0);
        F = bottom[0]->count(1);
        P = this->layer_param_.margin_softmax_loss_param().num_output();
        //P = this->layer_param_.inner_product_param().num_output();
        //lambda = this->layer_param_.margin_softmax_loss_param().lambda();
        this->lambda_.init(this->layer_param_.margin_softmax_loss_param().lambda());
        //printf("N=%d, F=%d, P=%d, lambda=%f\n", N, F, P, lambda);
        LOG(INFO) << "N=" << N << ",F=" << F << ",P=" << P << this->lambda_;

        this->blobs_.resize(1);
        vector<int> weight_shape(2);
        weight_shape[0] = P;
        weight_shape[1] = F;
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        // fill the weights
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.margin_softmax_loss_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());

        weight_shape[0] = std::max(N, (N*4 + P + P-1) / P);
        weight_shape[1] = P;
        prob_.Reshape(weight_shape);

        // initialize a column vecter of ones for GPU mode
        if (Caffe::mode() == Caffe::GPU) {
            //caffe_gpu_set(P, Dtype(1), prob_.mutable_gpu_diff() + N + N + P);
            caffe_set(P, Dtype(1), prob_.mutable_cpu_diff() + N*3);
        }

        weight_shape.resize(1);
        weight_shape[0] = N;
        ip_.Reshape(weight_shape);
    }
}

// no need to take 'sqrt'!!!
template <typename Dtype>
static Dtype caffe_cpu_norm2(const int n, const Dtype* x);
template <>
float caffe_cpu_norm2<float>(const int n, const float* x) {
   return cblas_snrm2(n, x, 1);
}
template <>
double caffe_cpu_norm2<double>(const int n, const double* x) {
    return cblas_dnrm2(n, x, 1);
}

template <typename Dtype>
void MarginSoftmaxLossLayer<Dtype>::Forward_cpu(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
    const Dtype* x = bottom[0]->cpu_data();

    const Dtype* w = this->blobs_[0]->cpu_data();
    //const Dtype* b = this->blobs_[1].cpu_data();

    Dtype *ip_data = ip_.mutable_cpu_data();
    Dtype *phi_data = ip_.mutable_cpu_diff();
    Dtype *prob_data = prob_.mutable_cpu_data();

    Dtype* x_norm = prob_.mutable_cpu_diff();
    Dtype* w_norm = x_norm + N;

    const Dtype* label = bottom[1]->cpu_data();

    /*for (int j = 0; j < P; j++) {
        w_norm[j] = caffe_cpu_norm2(F, w + j*F);
            //sqrt(caffe_cpu_dot(F, w + j*F, w + j*F));
    }*/

    Dtype loss = 0;

    static const Dtype cos_k_table[5] = { 1, M_SQRT1_2, 0, -M_SQRT1_2, -1 };

    const Dtype lambda = this->lambda_.get_iter("lambda");
    
    for (int i = 0; i < N; ++i) {
        const Dtype * const xi = x + i * F;
        Dtype x_norm_i = caffe_cpu_norm2(F, xi); //sqrt(caffe_cpu_dot(F, xi, xi));
        x_norm[i] = x_norm_i;
        const int yi = label[i];

        w_norm[i] = caffe_cpu_norm2(F, w + yi*F);

        Dtype max_fj = -FLT_MAX;

        for (int j = 0; j < P; j++) {
            Dtype fj = caffe_cpu_dot(F, xi, w + j*F);
            if (j == yi) {
                ip_data[i] = fj;

                Dtype xw = x_norm_i * w_norm[yi];
                Dtype cos_th = xw == 0 ? 1 : fj / xw;
                for (int k = 0; k < 4; ++k) {
                    //if (cos(k*M_PI / 4) >= cos_th && cos_th >= cos((k + 1)*M_PI / 4)) {
                    if (cos_k_table[k] >= cos_th && cos_th >= cos_k_table[k+1]) {
                        // (-1)^k * cos(m*theta) - 2k
                        // cos(2t) = 2cos(t)^2-1
                        // cos(3t) = 4cos(t)^3-3cos(t) = cos(t) * (4cos(t)^2 - 3)
                        // cos(4t) = 8cos(t)^4-8cos(t)^2+1 = 8cos(t)^2 * (cos(t)^2-1) + 1
                        Dtype c2 = cos_th * cos_th;
                        Dtype phi = (k & 1 ? (-1) : 1) * (8 * c2 * (c2 - 1) + 1) - 2 * k;
                        phi_data[i] = phi;
                        fj = (fj * lambda + xw * phi) / (1 + lambda);
                        break;
                    }
                }
            }
            prob_data[i*P + j] = fj;
            //cos_data[j] = fj / (x_norm_i * w_norm[j]);
            max_fj = std::max(max_fj, fj);
        }
        
        caffe_add_scalar(P, -max_fj, prob_data + i*P);

        caffe_exp(P, prob_data + i*P, prob_data + i*P);
        
        Dtype sum_exp = caffe_cpu_asum(P, prob_data + i*P);

        caffe_scal(P, 1 / sum_exp, prob_data + i*P);

        //for (int j = 0; j < P; ++j) printf("%f ", prob_data[i*P + j]);
        //printf("\n");

        loss -= log(prob_data[i*P + yi]);
    }
    loss /= N;
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MarginSoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype loss_weight = top[0]->cpu_diff()[0] / N;

    const Dtype* x = bottom[0]->cpu_data();
    Dtype* x_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(N*F, (Dtype)0.0, x_diff);

    const Dtype* w = this->blobs_[0]->cpu_data();
    Dtype *w_diff = this->blobs_[0]->mutable_cpu_diff();
    //caffe_set(P*F, (Dtype)0.0, w_diff);   // !!! should we do this???
    //printf("w_diff[0]=%f\n", w_diff[0]);

    //const Dtype* b = this->blobs_[1].cpu_data();

    Dtype *ip_data = ip_.mutable_cpu_data();
    Dtype *phi_data = ip_.mutable_cpu_diff();
    const Dtype *prob_data = prob_.cpu_data();
    const Dtype * const x_norm = prob_.cpu_diff();
    const Dtype * const w_norm = x_norm + N;

    const Dtype* label = bottom[1]->cpu_data();

    const Dtype lambda = this->lambda_.get();

    for (int i = 0; i < N; ++i) {
        const Dtype * const xi = x + i * F;
        const int yi = label[i];
        Dtype *x_diff_i = x_diff + i*F;
        const Dtype x_norm_i = x_norm[i];
        const Dtype ip = ip_data[i];
        const Dtype phi = phi_data[i];

        for (int j = 0; j < P; ++j) {
            const Dtype * const wj = w + j*F;
            Dtype *w_diff_j = w_diff + j*F;
            //const Dtype w_norm_j = w_norm[j];
            const Dtype prob_ij = prob_data[i*P + j];

            if (j == yi) {
                const Dtype w_norm_j = w_norm[yi];

                if (w_norm_j == 0) {
                    //printf("1\n");
                    caffe_axpy(F, loss_weight * (prob_ij - 1) * 4, xi, w_diff_j);
                }
                else if (x_norm_i == 0) {
                    //printf("2\n");
                    caffe_axpy(F, loss_weight * (prob_ij - 1) * 4, wj, x_diff_i);
                }
                else {
                    Dtype m = ( (-1 > phi && phi >= -3) || (-5 > phi && phi >= -7) ) * (-8);
                    m = m * (2 * ip*ip / (w_norm_j*w_norm_j*x_norm_i*x_norm_i) - 1);

                    // d(Li)/d(xi) = -d(f_yi)/d(xi) * ( 1-p(yi|xi,w) ) + \sum_{j!=y_i} w_j * p(j|wi,w)
                    //   d( f_yi = |w||x|phi(th) ) / d(x) = 
                    caffe_axpy(F,
                        loss_weight * (prob_ij - 1) * (
                            w_norm_j * phi / x_norm_i
                            - m * 2 * ip*ip / (x_norm_i*x_norm_i*x_norm_i*w_norm_j)
                            ) / (1 + lambda),
                        xi, x_diff_i);
                    caffe_axpy(F, loss_weight * (prob_ij - 1) *
                        (m*2 * ip / (x_norm_i * w_norm_j) + lambda) / (1 + lambda),
                        wj, x_diff_i);

                    // d(Li)/d(w_yi) = -d(f_yi)/d(w_yi) * ( 1-p(yi|xi,w) )
                    //   d( f_yi = |w||x|phi(th) ) / d(w) = 
                    caffe_axpy(F,
                        loss_weight * (prob_ij - 1) * (
                            x_norm_i * phi / w_norm_j
                            - m * 2 * ip*ip / (w_norm_j*w_norm_j*w_norm_j*x_norm_i)
                            ) / (1 + lambda),
                        wj, w_diff_j);
                    caffe_axpy(F, loss_weight * (prob_ij - 1) * 
                        (m*2 * ip / (x_norm_i * w_norm_j) + lambda) / (1 + lambda),
                        xi, w_diff_j);
                }
            }
            else {
                caffe_axpy(F, loss_weight * prob_ij, xi, w_diff_j);
                caffe_axpy(F, loss_weight * prob_ij, wj, x_diff_i);
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(MarginSoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(MarginSoftmaxLossLayer);
REGISTER_LAYER_CLASS(MarginSoftmaxLoss);

}  // namespace caffe
