#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void reduction_min_kernel(const int nthreads, const int N, const Dtype *data, Dtype *buf) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        if (index == nthreads - 1 && (N & 1))
            buf[index] = data[index << 1];
        else
            buf[index] = data[(index << 1)] < data[(index << 1) + 1] ? data[(index << 1)] : data[(index << 1) + 1];
    }
}
template <typename Dtype>
static void reduction_min(int N, const Dtype *data, Dtype *buf, Dtype *top) {
    int n = (N + 1) >> 1;
    reduction_min_kernel<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, N, data, buf);
    Dtype *buf2 = buf + n;
    while (N > 1) {
        N = n;
        n = (N + 1) >> 1;
        reduction_min_kernel<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, N, buf, buf2);
        std::swap(buf, buf2);
    }
    //caffe_gpu_axpby(1, Dtype(1), buf, Dtype(0), top);
    caffe_gpu_memcpy(sizeof(Dtype), buf, top);
}
template <typename Dtype>
static __global__ void reduction_max_kernel(const int nthreads, const int N, const Dtype *data, Dtype *buf) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        if (index == nthreads - 1 && (N & 1))
            buf[index] = data[index << 1];
        else
            buf[index] = data[(index << 1)] > data[(index << 1) + 1] ? data[(index << 1)] : data[(index << 1) + 1];
    }
}
template <typename Dtype>
static void reduction_max(int N, const Dtype *data, Dtype *buf, Dtype *top) {
    int n = (N + 1) >> 1;
    reduction_max_kernel<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, N, data, buf);
    Dtype *buf2 = buf + n;
    while (N > 1) {
        N = n;
        n = (N + 1) >> 1;
        reduction_max_kernel<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, N, buf, buf2);
        std::swap(buf, buf2);
    }
    //caffe_gpu_axpby(1, Dtype(1), buf, Dtype(0), top);
    caffe_gpu_memcpy(sizeof(Dtype), buf, top);
}

template <typename Dtype>
void ReductionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* mult_data = NULL;
  if (sum_multiplier_.count() > 0) {
    mult_data = sum_multiplier_.mutable_gpu_data();
  }
  Dtype* top_data = (op_ == ReductionParameter_ReductionOp_MIN ||
    op_ == ReductionParameter_ReductionOp_MAX) ? top[0]->mutable_gpu_data() : top[0]->mutable_cpu_data();
  for (int i = 0; i < num_; ++i) {
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_gpu_dot(dim_, mult_data, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_gpu_asum(dim_, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_gpu_dot(dim_, bottom_data, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_MIN:
      //{const Dtype *d = bottom[0]->cpu_data();
      //std::cerr << d[0];}
      reduction_min(dim_, bottom_data, mult_data, top_data);
      break;
    case ReductionParameter_ReductionOp_MAX:
      reduction_max(dim_, bottom_data, mult_data, top_data);
      break;
    default:
      LOG(FATAL) << "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
    }
    bottom_data += dim_;
    ++top_data;
  }
  if (coeff_ != Dtype(1)) {
    // Reset the top_data pointer.
    top_data = top[0]->mutable_gpu_data();
    caffe_gpu_scal(num_, coeff_, top_data);
  }
}

template <typename Dtype>
void ReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  // Get bottom_data, if needed.
  const Dtype* bottom_data = NULL;
  switch (op_) {
  // Operations that don't need bottom_data
  case ReductionParameter_ReductionOp_SUM:
  case ReductionParameter_ReductionOp_MEAN:
    break;
  // Operations that need bottom_data
  case ReductionParameter_ReductionOp_ASUM:
  case ReductionParameter_ReductionOp_SUMSQ:
    bottom_data = bottom[0]->gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  for (int i = 0; i < num_; ++i) {
    const Dtype bottom_coeff = (*top_diff) * coeff_;
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_gpu_set(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_gpu_sign(dim_, bottom_data, bottom_diff);
      caffe_gpu_scal(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_gpu_scale(dim_, 2 * bottom_coeff, bottom_data, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
    }
    bottom_data += dim_;
    bottom_diff += dim_;
    ++top_diff;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReductionLayer);

}  // namespace caffe
