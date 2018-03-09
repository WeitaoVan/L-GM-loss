#include <algorithm>
#include <vector>

#include "caffe/layers/margin_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void forward(const int nthreads, const int N_,
        const Dtype *data, const Dtype *label, const Dtype margin_, Dtype *data_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / N_;
        const int j = index % N_;
        const int y = (int)label[i];
        /*if (j != y)
            data_diff[index] = max(Dtype(0), margin_ + data[index]);
        else
            data_diff[index] = -data[index];*/
        if (j != y)
            data_diff[index] = max(Dtype(0), margin_ + data[index] - data[i*N_ + y]);
        else
            data_diff[index] = 0;
    }    
}
template <typename Dtype>
void MarginContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = M_*N_;
  forward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
        count, N_, bottom_data, label, margin_, bottom_diff);
  caffe_gpu_asum(M_*N_, bottom_diff, top[0]->mutable_cpu_data());
  top[0]->mutable_cpu_data()[0] /= M_;
}


__device__ inline void atomic_add(float * address, float val) {
    atomicAdd(address, val);
}
__device__ inline void atomic_add(double * address, double val) {
    unsigned long long int* address_as_ull =
            (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}

template <typename Dtype>
static __global__ void backward(const int nthreads, const int N_, const Dtype *label, Dtype *data_diff, const Dtype scale) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / N_;
        const int j = index % N_;
        const int y = (int)label[i];
        /*if (j != y) {
            if (data_diff[index] > 0)
                data_diff[index] = scale;
            else
                data_diff[index] = 0;
        }else {
            data_diff[index] = -scale;
        }*/
        if (data_diff[index] > 0) {
            data_diff[index] = scale; //-scale;
            atomic_add(&data_diff[i*N_ + y], -scale); //scale);
        }else if (j != y) {
            data_diff[index] = 0;
        }
    }    
}
template <typename Dtype>
void MarginContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* label = bottom[1]->gpu_data();
  const int count = M_*N_;
  Dtype scale = Dtype(1) / M_ * top[0]->cpu_diff()[0];
  backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
        count, N_, label, bottom_diff, scale);
}

INSTANTIATE_LAYER_GPU_FUNCS(MarginContrastiveLossLayer);

}  // namespace caffe
