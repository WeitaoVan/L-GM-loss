#include <vector>

#include "caffe/layers/gather_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void gather(const int nthreads, const int K, const Dtype* bottom_data,
    const Dtype* idx, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = bottom_data[index * K + (int)idx[index]];
  }
}

template <typename Dtype>
__global__ void gather_inverse(const int nthreads, const int K, const Dtype* bottom_data,
    const Dtype* idx, Dtype* top_data) {
  // note: K is the second dimension of the OUTPUT, not the input.
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / K;
    const int j = index % K;
    if (j < (int)idx[i])
      top_data[index] = bottom_data[i * (K+1) + j];
    else
      top_data[index] = bottom_data[i * (K+1) + j + 1];
  }
}

template <typename Dtype>
void GatherLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const GatherParameter& param = this->layer_param_.gather_param();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* idx = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const bool inverse = param.inverse();
  const int M = bottom[0]->count(0, 1);
  const int K = bottom[0]->count(1, 2);
  CHECK_EQ(M * K, bottom[0]->count()) << "M,K=" << M << ","<< K << ", only support 2-d bottom[0], e.g. the output of fc layer";
  const int nthreads = inverse? M*(K-1): M;
  if (!inverse){
    gather<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, K, bottom_data, idx, top_data);
  }
  else {
    gather_inverse<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, K-1, bottom_data, idx, top_data);
  }

}

template <typename Dtype>
void GatherLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(GatherLayer);

}  // namespace caffe
