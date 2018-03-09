//#include <algorithm>
#include <cfloat>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/iou_loss.hpp"
#include "caffe/util/math_functions.hpp"

using std::min;
using std::abs;

namespace caffe {

template <typename Dtype>
void IoULossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void IoULossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    CHECK_EQ(bottom[0]->shape_string() == bottom[1]->shape_string(), true) << "Two bottom blobs should have the same shape";
    CHECK_EQ(bottom[0]->shape(1), 4) << "Shape(1) shoud equal 4";

    N = bottom[0]->shape(0);
    inner_size = bottom[0]->count(2);

    LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void IoULossLayer<Dtype>::Forward_cpu(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
    const Dtype* x = bottom[0]->cpu_data();
    const Dtype* xg = bottom[1]->cpu_data();

    Dtype loss = 0;

    // batch size = N
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < inner_size; ++j) {
            Dtype tg = xg[(i * 4 + 0) * inner_size + j];
            Dtype bg = xg[(i * 4 + 1) * inner_size + j];

            if (tg + bg == -1)
                continue;

            Dtype lg = xg[(i * 4 + 2) * inner_size + j];
            Dtype rg = xg[(i * 4 + 3) * inner_size + j];

            Dtype t = x[(i * 4 + 0) * inner_size + j];
            Dtype b = x[(i * 4 + 1) * inner_size + j];
            Dtype l = x[(i * 4 + 2) * inner_size + j];
            Dtype r = x[(i * 4 + 3) * inner_size + j];

            Dtype A = (b + t + 1) * (r + l + 1);
            Dtype Ag = (bg + tg + 1) * (rg + lg + 1);
            Dtype I = (min(b, bg) + min(t, tg) + 1) * (min(r, rg) + min(l, lg) + 1);
            //if (I <= 1e-4) I = 1e-4;

            // use L1 loss if I is too small (or even negative)
            if (I <= 1e-4) {
                loss -= abs(t - tg) + abs(b - bg) + abs(l - lg) + abs(r - rg);
            }
            else {
                Dtype U = A + Ag - I;
                loss -= log(I / U);
            }
            //if (std::isnan(loss)) {
            //    printf("%f %f %f\n", loss, I, U);
            //}
        }
    }
    loss /= N;
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void IoULossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype loss_weight = top[0]->cpu_diff()[0] / N;

    const Dtype* x = bottom[0]->cpu_data();
    const Dtype* xg = bottom[1]->cpu_data();
    Dtype* x_diff = bottom[0]->mutable_cpu_diff();
    //caffe_set(N*F, (Dtype)0.0, x_diff);

    // batch size = N
    for (int i = 0; i < N; ++i) {
        // inner size = W * H
        for (int j = 0; j < inner_size; ++j) {
            //
            // L = -ln(I/U) = ln(U) - ln(I)
            //
            // dL/dt = 1/U * (dA/dt - dI/dt) - 1/I * dI/dt = 1/U * dA/dt - (1/U + 1/A) * dI/dt
            //   dA/dt = d[(b+t+1)*(r+l+1)]/dt = r+l+1
            //   dI/dt = min(r,rg)+min(l,lg)+1, if t < tg

            Dtype tg = xg[(i * 4 + 0) * inner_size + j];
            Dtype bg = xg[(i * 4 + 1) * inner_size + j];

            if (tg + bg == -1) {
                x_diff[(i * 4 + 0) * inner_size + j] = 0;
                x_diff[(i * 4 + 1) * inner_size + j] = 0;
                x_diff[(i * 4 + 2) * inner_size + j] = 0;
                x_diff[(i * 4 + 3) * inner_size + j] = 0;
                continue;
            }

            Dtype lg = xg[(i * 4 + 2) * inner_size + j];
            Dtype rg = xg[(i * 4 + 3) * inner_size + j];

            Dtype t = x[(i * 4 + 0) * inner_size + j];
            Dtype b = x[(i * 4 + 1) * inner_size + j];
            Dtype l = x[(i * 4 + 2) * inner_size + j];
            Dtype r = x[(i * 4 + 3) * inner_size + j];

            Dtype h = (b + t + 1), w = (r + l + 1);
            Dtype A = w * h;
            Dtype Ag = (bg + tg + 1) * (rg + lg + 1);
            Dtype hI = (min(b, bg) + min(t, tg) + 1), wI = (min(r, rg) + min(l, lg) + 1);
            Dtype I = hI * wI;

            // L1 loss
            if (I < 1e-4) {
                x_diff[(i * 4 + 0) * inner_size + j] = loss_weight * ((t > tg) - (t < tg));
                x_diff[(i * 4 + 1) * inner_size + j] = loss_weight * ((b > bg) - (b < bg));
                x_diff[(i * 4 + 2) * inner_size + j] = loss_weight * ((l > lg) - (l < lg));
                x_diff[(i * 4 + 3) * inner_size + j] = loss_weight * ((r > rg) - (r < rg));
            }
            else {
                Dtype U = 1 / (A + Ag - I);
                I = 1 / I;

                x_diff[(i * 4 + 0) * inner_size + j] = loss_weight * (U * w - (t < tg) * (U + I) * wI);
                x_diff[(i * 4 + 1) * inner_size + j] = loss_weight * (U * w - (b < bg) * (U + I) * wI);
                x_diff[(i * 4 + 2) * inner_size + j] = loss_weight * (U * h - (l < lg) * (U + I) * hI);
                x_diff[(i * 4 + 3) * inner_size + j] = loss_weight * (U * h - (r < rg) * (U + I) * hI);
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(IoULossLayer);
#endif

INSTANTIATE_CLASS(IoULossLayer);
REGISTER_LAYER_CLASS(IoULoss);

}  // namespace caffe
