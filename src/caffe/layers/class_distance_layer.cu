#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/class_distance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//////////////////////////////////////////
template <typename Dtype>
static __global__ void compute_top_l2(const int nthreads, const int N_, const int K_,
    const Dtype *bottom_data, Dtype *top_data, const Dtype *weight, const Dtype *sigma, 
    Dtype *dist, bool ignore_zero, bool isotropic) {

    if (ignore_zero) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int i = index / N_;
            const int j = index % N_;
            Dtype t = 0;
            for (int k = 0; k < K_; ++k) if (bottom_data[i*K_ + k]) {
                Dtype d = weight[j*K_ + k] - bottom_data[i*K_ + k];
                t += d*d;
            }
	    dist[index] = t;
            top_data[index] = Dtype(-0.5) * t / (max(sigma[j], Dtype(0)) + Dtype(0.0001));
        }
    }
    else {
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int i = index / N_;
            const int j = index % N_;
            Dtype t = 0;
            for (int k = 0; k < K_; ++k) {
                Dtype d = weight[j*K_ + k] - bottom_data[i*K_ + k];
		if (isotropic)
                  t += d*d;
		else
   		  t += d*d/max(sigma[j*K_ + k], Dtype(0) + Dtype(0.00000001));
            }
	    dist[index] = t; // only useful for 'isotropic'
	    if (isotropic)
              top_data[index] = Dtype(-0.5) * t / (max(sigma[j], Dtype(0)) + Dtype(0.0001));
	    else
	      top_data[index] = Dtype(-0.5) * t;
        }
    }
    
}

template <typename Dtype>
static __global__ void compute_top_ip(const int nthreads, const int N_, const int K_,
    const Dtype *bottom_data, Dtype *top_data, const Dtype *weight) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / N_;
        const int j = index % N_;
        Dtype t = 0;
        for (int k = 0; k < K_; ++k) {
            t += weight[j*K_ + k] * bottom_data[i*K_ + k];
        }
        top_data[index] = t;
    }
}

template <typename Dtype>
static __global__ void compute_top_l1(const int nthreads, const int N_, const int K_,
    const Dtype *bottom_data, Dtype *top_data, const Dtype *weight, bool ignore_zero) {

    if (ignore_zero) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int i = index / N_;
            const int j = index % N_;
            Dtype t = 0;
            for (int k = 0; k < K_; ++k) {
                Dtype d = weight[j*K_ + k] - bottom_data[i*K_ + k];
                t += abs(d);
            }
            top_data[index] = -t;
        }
    }
    else {
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int i = index / N_;
            const int j = index % N_;
            Dtype t = 0;
            for (int k = 0; k < K_; ++k) {
                Dtype d = weight[j*K_ + k] - bottom_data[i*K_ + k];
                t += abs(d);
            }
            top_data[index] = -t;
        }
    }
}

//////////////////////////////////////////
template <typename Dtype>
static __global__ void margin_top(const int M_, const int N_,
    Dtype *top_data, const Dtype *label, const Dtype margin_mul, const Dtype margin_add) {


    CUDA_KERNEL_LOOP(i, M_) {
        const int y = (int)label[i];
        top_data[i*N_ + y] += top_data[i*N_ + y] * margin_mul - margin_add;
        
    }
}
//////////////////////////////////////////

template <typename Dtype>
void ClassDistanceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    const Dtype* sigma = this->blobs_[1]->gpu_data();
    Dtype* dist = dist_.mutable_gpu_data();

    const ClassDistanceParameter& param = this->layer_param_.class_distance_param();
    bool isotropic = param.isotropic();

    switch (param.metric()) {
    case ClassDistanceParameter_Metric_L2:
        compute_top_l2<Dtype> << <CAFFE_GET_BLOCKS(M_*N_), CAFFE_CUDA_NUM_THREADS >> >(
            M_*N_, N_, K_, bottom_data, top_data, weight, sigma, dist, param.ignore_zero() & (this->phase_ == TRAIN), 
            isotropic);
        break;
    case ClassDistanceParameter_Metric_IP:
        compute_top_ip<Dtype> << <CAFFE_GET_BLOCKS(M_*N_), CAFFE_CUDA_NUM_THREADS >> >(
            M_*N_, N_, K_, bottom_data, top_data, weight);
        break;
    case ClassDistanceParameter_Metric_L1:
        compute_top_l1<Dtype> << <CAFFE_GET_BLOCKS(M_*N_), CAFFE_CUDA_NUM_THREADS >> >(
            M_*N_, N_, K_, bottom_data, top_data, weight, param.ignore_zero() & (this->phase_ == TRAIN));
        break;
    }
  
    if (bottom.size() == 2 && this->phase_ == TRAIN) {
        Dtype margin_mul_ = this->m_mul_.get_iter("mul_margin");
        Dtype margin_add_ = this->m_add_.get_iter("add_margin");

        const Dtype* label = bottom[1]->gpu_data();

        margin_top<Dtype> << <CAFFE_GET_BLOCKS(M_), CAFFE_CUDA_NUM_THREADS >> >(
            M_, N_, top_data, label, margin_mul_, margin_add_);
    }

    // validate that sigma > 0
    const Dtype *sigma_cpu = this->blobs_[1]->cpu_data();
    const int sigma_number = isotropic?N_:(N_*K_);
    for(int i=0; i<sigma_number; i++)
	if (sigma_cpu[i] <= eps_) {
	  LOG(INFO) << "Dangerous sigma value, sigma[" << i << "]=" << sigma_cpu[i];
	  break;
        }

    /*if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
            bias_multiplier_.gpu_data(),
            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
    }*/
}


//==========================================

template <typename Dtype>
static __global__ void compute_gradient_bottom_label_l2(const int nthreads, const int K_, const int N_,
    const Dtype* top_diff, const Dtype *bottom_data, Dtype *bottom_diff, const Dtype *weight,
    const Dtype *label, const Dtype margin_mul, const Dtype center_coef, const Dtype *sigma, bool ignore_zero) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int j = 0; j < N_; ++j) if (!ignore_zero || bottom_data[index]) {
            if (j == (int)label[i])
                t += (weight[j*K_ + k] - bottom_data[index]) * (margin_mul / (max(sigma[j], Dtype(0)) + Dtype(0.0001)) * top_diff[i*N_ + j] - center_coef);
            else
                t += (weight[j*K_ + k] - bottom_data[index]) / (max(sigma[j], Dtype(0)) + Dtype(0.0001)) * top_diff[i*N_ + j];
        }
        bottom_diff[index] = t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_weight_label_l2(const int nthreads, const int K_, const int M_,
    const Dtype* top_diff, const Dtype *bottom_data, const Dtype *weight, Dtype* weight_diff,
    const Dtype* label, const Dtype margin_mul, const Dtype center_coef, const Dtype *sigma, 
    Dtype *sigma_diff, const Dtype *dist, bool update_sigma, bool ignore_zero) {

    const int N_ = nthreads / K_;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int j = index / K_;
        const int k = index % K_;
        Dtype t = 0;
	Dtype t_sigma = 0;
        for (int i = 0; i < M_; ++i) if (!ignore_zero || bottom_data[i*K_ + k]) {
            if (j == (int)label[i]){
                t += (bottom_data[i*K_ + k] - weight[index]) * (margin_mul / (max(sigma[j], Dtype(0)) + Dtype(0.0001)) * top_diff[i*N_ + j] - center_coef);
		if (update_sigma && k==0)
		  t_sigma += dist[i * N_ + j] * margin_mul / (Dtype(2.0) * (max(sigma[j], Dtype(0)) + Dtype(0.0001)) * sigma[j]) * top_diff[i*N_ + j];
	    }
            else{
                t += (bottom_data[i*K_ + k] - weight[index]) / (max(sigma[j], Dtype(0)) + Dtype(0.0001)) * top_diff[i*N_ + j];
		if (update_sigma && k==0)
		  t_sigma += dist[i * N_ + j] / (Dtype(2.0) * (max(sigma[j], Dtype(0)) + Dtype(0.0001)) * sigma[j]) * top_diff[i*N_ + j];
	    }
        }
        weight_diff[index] += t;
	if (update_sigma && k == 0)
	  sigma_diff[j] += t_sigma;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_bottom_label_l2_diag(const int nthreads, const int K_, const int N_,
    const Dtype* top_diff, const Dtype *bottom_data, Dtype *bottom_diff, const Dtype *weight,
    const Dtype *label, const Dtype margin_mul, const Dtype center_coef, const Dtype *sigma, bool ignore_zero) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int j = 0; j < N_; ++j) if (!ignore_zero || bottom_data[index]) {
            if (j == (int)label[i])
                t += (weight[j*K_ + k] - bottom_data[index]) * (margin_mul / (max(sigma[j*K_ + k] , Dtype(0)) + Dtype(0.00000001)) * top_diff[i*N_ + j] - center_coef);
            else
                t += (weight[j*K_ + k] - bottom_data[index]) / (max(sigma[j*K_ + k], Dtype(0)) + Dtype(0.00000001)) * top_diff[i*N_ + j];
        }
        bottom_diff[index] = t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_weight_label_l2_diag(const int nthreads, const int K_, const int M_,
    const Dtype* top_diff, const Dtype *bottom_data, const Dtype *weight, Dtype* weight_diff,
    const Dtype* label, const Dtype margin_mul, const Dtype center_coef, const Dtype *sigma, 
    Dtype *sigma_diff, bool update_sigma, bool ignore_zero) {

    const int N_ = nthreads / K_;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int j = index / K_;
        const int k = index % K_;
        Dtype t = 0;
	Dtype t_sigma = 0;
        Dtype d = 0;
	Dtype safe_sigma = max(sigma[index], Dtype(0)) + Dtype(0.0001);
        for (int i = 0; i < M_; ++i) if (!ignore_zero || bottom_data[i*K_ + k]) {
	    d = bottom_data[i*K_ + k] - weight[index];
            if (j == (int)label[i]){
                t += d * (margin_mul / safe_sigma * top_diff[i*N_ + j] - center_coef);
		if (update_sigma)
		  t_sigma += d * d * margin_mul / (Dtype(2.0) * safe_sigma * safe_sigma) * top_diff[i*N_ + j];
	    }
            else{
                t += d / safe_sigma * top_diff[i*N_ + j];
		if (update_sigma)
		  t_sigma += d * d / (Dtype(2.0) * safe_sigma * safe_sigma) * top_diff[i*N_ + j];
	    }
        }
        weight_diff[index] += t;
	if (update_sigma)
	  sigma_diff[index] += t_sigma;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_bottom_label_ip(const int nthreads, const int K_, const int N_,
    const Dtype* top_diff, const Dtype *bottom_data, Dtype *bottom_diff, const Dtype *weight,
    const Dtype *label, const Dtype margin_mul, const Dtype center_coef) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int j = 0; j < N_; ++j) {
            if (j == (int)label[i])
                t += weight[j*K_ + k] * margin_mul * top_diff[i*N_ + j] +
                    (bottom_data[index] - weight[j*K_ + k]) * center_coef;
            else
                t += weight[j*K_ + k] * top_diff[i*N_ + j];
        }
        bottom_diff[index] = t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_weight_label_ip(const int nthreads, const int K_, const int M_,
    const Dtype* top_diff, const Dtype *bottom_data, const Dtype *weight, Dtype* weight_diff,
    const Dtype* label, const Dtype margin_mul, const Dtype center_coef) {

    const int N_ = nthreads / K_;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int j = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int i = 0; i < M_; ++i) {
            if (j == (int)label[i])
                t += margin_mul * bottom_data[i*K_ + k] * top_diff[i*N_ + j] +
                    (weight[index] - bottom_data[i*K_ + k]) * center_coef;
            else
                t += bottom_data[i*K_ + k] * top_diff[i*N_ + j];
        }
        weight_diff[index] += t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_bottom_label_l1(const int nthreads, const int K_, const int N_,
    const Dtype* top_diff, const Dtype *bottom_data, Dtype *bottom_diff, const Dtype *weight,
    const Dtype *label, const Dtype margin_mul, const Dtype center_coef, bool ignore_zero) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int j = 0; j < N_; ++j) if (!ignore_zero || bottom_data[index]) {
            if (j == (int)label[i])
                t += ((weight[j*K_ + k] > bottom_data[index]) - (bottom_data[index] > weight[j*K_ + k]))
                    * (margin_mul * top_diff[i*N_ + j] - center_coef);
            else
                t += ((weight[j*K_ + k] > bottom_data[index]) - (bottom_data[index] > weight[j*K_ + k])) * top_diff[i*N_ + j];
        }
        bottom_diff[index] = t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_weight_label_l1(const int nthreads, const int K_, const int M_,
    const Dtype* top_diff, const Dtype *bottom_data, const Dtype *weight, Dtype* weight_diff,
    const Dtype* label, const Dtype margin_mul, const Dtype center_coef, bool ignore_zero) {

    const int N_ = nthreads / K_;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int j = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int i = 0; i < M_; ++i) if (!ignore_zero || bottom_data[i*K_ + k]) {
            if (j == (int)label[i])
                t += ((bottom_data[i*K_ + k] - weight[index]) - (weight[index] > bottom_data[i*K_ + k]) - (bottom_data[index] > weight[j*K_ + k]))
                    * (margin_mul * top_diff[i*N_ + j] - center_coef);
            else
                t += ((bottom_data[i*K_ + k] - weight[index]) - (weight[index] > bottom_data[i*K_ + k]) - (bottom_data[index] > weight[j*K_ + k])) * top_diff[i*N_ + j];
        }
        weight_diff[index] += t;
    }
}


template <typename Dtype>
static __global__ void compute_gradient_bottom_l2(const int nthreads, const int K_, const int N_,
    const Dtype* top_diff, const Dtype *bottom_data, Dtype *bottom_diff, const Dtype *weight, bool ignore_zero) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int j = 0; j < N_; ++j) if (!ignore_zero || bottom_data[index])
            t += (weight[j*K_ + k] - bottom_data[index]) * top_diff[i*N_ + j];
        bottom_diff[index] = t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_weight_l2(const int nthreads, const int K_, const int M_,
    const Dtype* top_diff, const Dtype *bottom_data, const Dtype *weight, Dtype* weight_diff, bool ignore_zero) {

    const int N_ = nthreads / K_;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int j = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int i = 0; i < M_; ++i) if (!ignore_zero || bottom_data[i*K_ + k])
            t += (bottom_data[i*K_ + k] - weight[index]) * top_diff[i*N_ + j];
        weight_diff[index] += t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_bottom_ip(const int nthreads, const int K_, const int N_,
    const Dtype* top_diff, const Dtype *bottom_data, Dtype *bottom_diff, const Dtype *weight) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int j = 0; j < N_; ++j)
            t += weight[j*K_ + k] * top_diff[i*N_ + j];
        bottom_diff[index] = t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_weight_ip(const int nthreads, const int K_, const int M_,
    const Dtype* top_diff, const Dtype *bottom_data, const Dtype *weight, Dtype* weight_diff) {

    const int N_ = nthreads / K_;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int j = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int i = 0; i < M_; ++i)
            t += bottom_data[i*K_ + k] * top_diff[i*N_ + j];
        weight_diff[index] += t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_bottom_l1(const int nthreads, const int K_, const int N_,
    const Dtype* top_diff, const Dtype *bottom_data, Dtype *bottom_diff, const Dtype *weight, bool ignore_zero) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int j = 0; j < N_; ++j) if (!ignore_zero || bottom_data[index])
            t += ((weight[j*K_ + k] > bottom_data[index]) - (bottom_data[index] > weight[j*K_ + k])) * top_diff[i*N_ + j];
        bottom_diff[index] = t;
    }
}

template <typename Dtype>
static __global__ void compute_gradient_weight_l1(const int nthreads, const int K_, const int M_,
    const Dtype* top_diff, const Dtype *bottom_data, const Dtype *weight, Dtype* weight_diff, bool ignore_zero) {

    const int N_ = nthreads / K_;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int j = index / K_;
        const int k = index % K_;
        Dtype t = 0;
        for (int i = 0; i < M_; ++i) if (!ignore_zero || bottom_data[i*K_ + k])
            t += ((bottom_data[i*K_ + k] - weight[index]) - (weight[index] > bottom_data[i*K_ + k]) - (bottom_data[index] > weight[j*K_ + k])) * top_diff[i*N_ + j];
        weight_diff[index] += t;
    }
}


template <typename Dtype>
void ClassDistanceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  /*const*/ Dtype* weight = this->blobs_[0]->mutable_gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* sigma = this->blobs_[1]->gpu_data();
  const Dtype* dist = dist_.gpu_data();
  Dtype* sigma_diff = this->blobs_[1]->mutable_gpu_diff();
  

  const ClassDistanceParameter& param = this->layer_param_.class_distance_param();
  bool ignore_zero = param.ignore_zero();
  bool update_sigma = param.update_sigma();
  bool isotropic = param.isotropic();

  if (isotropic)
    caffe_gpu_set(N_, (Dtype)0, sigma_diff);
  else
    caffe_gpu_set(N_*K_, (Dtype)0, sigma_diff);

  if (bottom.size() == 2) {
      const Dtype* label = bottom[1]->gpu_data();
      const Dtype center_coef_ = param.center_coef() / M_;

      const Dtype margin_mul_1 = 1 + (param.block_mul_grad() ? 0 : m_mul_.get());

      switch (param.metric()) {
      case ClassDistanceParameter_Metric_L2:
	  if (isotropic) {
            compute_gradient_bottom_label_l2<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
                M_*K_, K_, N_, top_diff, bottom_data, bottom_diff, weight, label, margin_mul_1, center_coef_, sigma, ignore_zero);
            compute_gradient_weight_label_l2<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
                N_*K_, K_, M_, top_diff, bottom_data, weight, weight_diff, label, margin_mul_1, center_coef_, sigma, sigma_diff, dist, update_sigma, ignore_zero);
	  }
	  else {
            compute_gradient_bottom_label_l2_diag<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
                M_*K_, K_, N_, top_diff, bottom_data, bottom_diff, weight, label, margin_mul_1, center_coef_, sigma, ignore_zero);
            compute_gradient_weight_label_l2_diag<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
                N_*K_, K_, M_, top_diff, bottom_data, weight, weight_diff, label, margin_mul_1, center_coef_, sigma, sigma_diff, update_sigma, ignore_zero);
	  }
          break;
      case ClassDistanceParameter_Metric_IP:
          compute_gradient_bottom_label_ip<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              M_*K_, K_, N_, top_diff, bottom_data, bottom_diff, weight, label, margin_mul_1, center_coef_);
          compute_gradient_weight_label_ip<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              N_*K_, K_, M_, top_diff, bottom_data, weight, weight_diff, label, margin_mul_1, center_coef_);
          break;
      case ClassDistanceParameter_Metric_L1:
          compute_gradient_bottom_label_l1<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              M_*K_, K_, N_, top_diff, bottom_data, bottom_diff, weight, label, margin_mul_1, center_coef_, ignore_zero);
          compute_gradient_weight_label_l1<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              N_*K_, K_, M_, top_diff, bottom_data, weight, weight_diff, label, margin_mul_1, center_coef_, ignore_zero);
          break;
      }
  }
  else {
      switch (param.metric()) {
      case ClassDistanceParameter_Metric_L2:
          compute_gradient_bottom_l2<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              M_*K_, K_, N_, top_diff, bottom_data, bottom_diff, weight, ignore_zero);
          compute_gradient_weight_l2<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              N_*K_, K_, M_, top_diff, bottom_data, weight, weight_diff, ignore_zero);
          break;
      case ClassDistanceParameter_Metric_IP:
          compute_gradient_bottom_ip<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              M_*K_, K_, N_, top_diff, bottom_data, bottom_diff, weight);
          compute_gradient_weight_ip<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              N_*K_, K_, M_, top_diff, bottom_data, weight, weight_diff);
          break;
      case ClassDistanceParameter_Metric_L1:
          compute_gradient_bottom_l1<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              M_*K_, K_, N_, top_diff, bottom_data, bottom_diff, weight, ignore_zero);
          compute_gradient_weight_l1<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
              N_*K_, K_, M_, top_diff, bottom_data, weight, weight_diff, ignore_zero);
          break;
      }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(ClassDistanceLayer);

}  // namespace caffe
