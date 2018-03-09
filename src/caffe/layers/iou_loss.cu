#include <cfloat>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include "caffe/layers/iou_loss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void forward_pixel(const int nthreads, const int inner_size,
    const Dtype* x, const Dtype* xg, Dtype* loss_buf) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        int i = index / inner_size, j = index % inner_size;

        Dtype tg = xg[(i * 4 + 0) * inner_size + j];
        Dtype bg = xg[(i * 4 + 1) * inner_size + j];
        
        if (tg + bg == -1) {
            loss_buf[index] = 0;
            continue;
        }
        
        Dtype lg = xg[(i * 4 + 2) * inner_size + j];
        Dtype rg = xg[(i * 4 + 3) * inner_size + j];
        
        Dtype t = x[(i * 4 + 0) * inner_size + j];
        Dtype b = x[(i * 4 + 1) * inner_size + j];
        Dtype l = x[(i * 4 + 2) * inner_size + j];
        Dtype r = x[(i * 4 + 3) * inner_size + j];

        Dtype A = (b + t + 1) * (r + l + 1);
        Dtype Ag = (bg + tg + 1) * (rg + lg + 1);
        Dtype I = (min(b, bg) + min(t, tg) + 1) * (min(r, rg) + min(l, lg) + 1);

        // use L1 loss if I is too small (or even negative)
        if (I < 1e-2) {
            loss_buf[index] = abs(t - tg) + abs(b - bg) + abs(l - lg) + abs(r - rg);
        }
        else {
            Dtype U = A + Ag - I;
            loss_buf[index] = -log(I / U);
        }
    }
}

template <typename Dtype>
void IoULossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* const x = bottom[0]->gpu_data();
    const Dtype* const xg = bottom[1]->gpu_data();
    Dtype* const loss_buf = bottom[0]->mutable_gpu_diff();

    int nthreads = N*inner_size;
    forward_pixel<Dtype> << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
        nthreads, inner_size,  x, xg,  loss_buf);

    // sum all losses together
    Dtype* const loss = top[0]->mutable_cpu_data();
    caffe_gpu_asum(nthreads, loss_buf, loss);
    //caffe_gpu_scal(1, Dtype(1) / N, loss);
    *loss /= N;
}

template <typename Dtype>
static __global__ void backward_pixel(const int nthreads, const int inner_size,
    const Dtype* x, const Dtype* xg, const Dtype loss_weight, Dtype* x_diff) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        int i = index / inner_size, j = index % inner_size;

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
        if (I < 1e-2) {
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

template <typename Dtype>
void IoULossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* const x = bottom[0]->gpu_data();
    const Dtype* const xg = bottom[1]->gpu_data();
    Dtype* const x_diff = bottom[0]->mutable_gpu_diff();

    const Dtype loss_weight = top[0]->cpu_diff()[0] / N;

    int nthreads = N*inner_size;
    backward_pixel<Dtype> << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
        nthreads, inner_size,  x, xg, loss_weight,  x_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(IoULossLayer);

}  // namespace caffe
