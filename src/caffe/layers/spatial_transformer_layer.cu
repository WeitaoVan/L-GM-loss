#include <algorithm>
#include <vector>
#include "cuda.h"
#include "caffe/layer.hpp"
#include "caffe/layers/spatial_transformer_layer.hpp"

namespace caffe {


///////////////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void forward_affine(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
		const Dtype* in, const Dtype* theta, Dtype* source_data, Dtype* out) {//, const Dtype* fill_value_) {

    //int div = channels_ * output_H_ * output_W_;
    const int map_size = output_H_ * output_W_;
    
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

        int offset = 6 * n;
        Dtype x = x_target * theta[offset] + y_target * theta[offset + 1] + theta[offset + 2];
        Dtype y = x_target * theta[offset + 3] + y_target * theta[offset + 4] + theta[offset + 5];

        x = (x + (Dtype) 1.) / 2 * (width_ - 1);
        y = (y + (Dtype) 1.) / 2 * (height_ - 1);

        //offset = n * map_size * 2 + h * output_W_ + w;
		offset = (n * map_size + h * output_W_ + w) * 2;
        source_data[offset] = x;
        //source_data[offset + map_size] = y;
		source_data[offset + 1] = y;

		x = x > 0 ? x : 0; x = x < (width_ - 1) ? x : width_ - 1;
		y = y > 0 ? y : 0; y = y < (height_ - 1) ? y : height_ - 1;
		int w_min = (int)floor(x);
		int w_max = (int)ceil(x);
		int h_min = (int)floor(y);
		int h_max = (int)ceil(y);
		for (int c = 0; c < channels_; ++c) {
			Dtype r = 0;
			offset = (n * channels_ + c) * height_ * width_;
			for (int hh = h_min; hh <= h_max; ++hh) {
				const Dtype dy = (1 - fabs(y - hh));
				for (int ww = w_min; ww <= w_max; ++ww) {
					r += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = r;
		}

        /*int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		for (int c = 0; c < channels_; ++c) {
			Dtype tmp;
			if (h_max < h_min || w_max < w_min) {
				tmp = fill_value_[c];
			}else {
				tmp = 0;
				offset = (n * channels_ + c) * height_ * width_;
				for (int hh = h_min; hh <= h_max; ++hh) {
					const Dtype dy = (1 - fabs(y - hh));
					for (int ww = w_min; ww <= w_max; ++ww) {
						tmp += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
					}
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = tmp;
		}*/
    }
}

template <typename Dtype>
__global__ void forward_translation(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* in, const Dtype* theta, Dtype* source_data, Dtype* out,
		const float theta_1_1, const float theta_2_2//, const Dtype* fill_value_
		) {

    const int map_size = output_H_ * output_W_;
    
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

        int offset = 2 * n;
		Dtype x = theta_1_1 * x_target + theta[offset];
        Dtype y = theta_2_2 * y_target + theta[offset + 1];

        x = (x + (Dtype) 1.) / 2 * (width_ - 1);
        y = (y + (Dtype) 1.) / 2 * (height_ - 1);

		offset = (n * map_size + h * output_W_ + w) * 2;
        source_data[offset] = x;
		source_data[offset + 1] = y;

		x = x > 0 ? x : 0; x = x < (width_ - 1) ? x : width_ - 1;
		y = y > 0 ? y : 0; y = y < (height_ - 1) ? y : height_ - 1;
		int w_min = (int)floor(x);
		int w_max = (int)ceil(x);
		int h_min = (int)floor(y);
		int h_max = (int)ceil(y);
		for (int c = 0; c < channels_; ++c) {
			Dtype r = 0;
			offset = (n * channels_ + c) * height_ * width_;
			for (int hh = h_min; hh <= h_max; ++hh) {
				const Dtype dy = (1 - fabs(y - hh));
				for (int ww = w_min; ww <= w_max; ++ww) {
					r += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = r;
		}
        /*int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		for (int c = 0; c < channels_; ++c) {
			Dtype tmp;
			if (h_max < h_min || w_max < w_min) {
				tmp = fill_value_[c];
			}
			else {
				tmp = 0;
				offset = (n * channels_ + c) * height_ * width_;
				for (int hh = h_min; hh <= h_max; ++hh) {
					const Dtype dy = (1 - fabs(y - hh));
					for (int ww = w_min; ww <= w_max; ++ww) {
						tmp += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
					}
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = tmp;
		}*/
    }
}

template <typename Dtype>
__global__ void forward_translation_scaling(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
		const Dtype* in, const Dtype* theta, Dtype* source_data, Dtype* out) {//, const Dtype* fill_value_) {

    const int map_size = output_H_ * output_W_;
    
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

        int offset = 4 * n;
		Dtype x = x_target * theta[offset] + theta[offset + 1];
        Dtype y = y_target * theta[offset + 2] + theta[offset + 3];

        x = (x + (Dtype) 1.) / 2 * (width_ - 1);
        y = (y + (Dtype) 1.) / 2 * (height_ - 1);

		offset = (n * map_size + h * output_W_ + w) * 2;
        source_data[offset] = x;
		source_data[offset + 1] = y;

		x = x > 0 ? x : 0; x = x < (width_ - 1) ? x : width_ - 1;
		y = y > 0 ? y : 0; y = y < (height_ - 1) ? y : height_ - 1;
		int w_min = (int)floor(x);
		int w_max = (int)ceil(x);
		int h_min = (int)floor(y);
		int h_max = (int)ceil(y);
		for (int c = 0; c < channels_; ++c) {
			Dtype r = 0;
			offset = (n * channels_ + c) * height_ * width_;
			for (int hh = h_min; hh <= h_max; ++hh) {
				const Dtype dy = (1 - fabs(y - hh));
				for (int ww = w_min; ww <= w_max; ++ww) {
					r += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = r;
		}
        /*int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		for (int c = 0; c < channels_; ++c) {
			Dtype tmp;
			if (h_max < h_min || w_max < w_min) {
				tmp = fill_value_[c];
			}
			else {
				tmp = 0;
				offset = (n * channels_ + c) * height_ * width_;
				for (int hh = h_min; hh <= h_max; ++hh) {
					const Dtype dy = (1 - fabs(y - hh));
					for (int ww = w_min; ww <= w_max; ++ww) {
						tmp += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
					}
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = tmp;
		}*/
    }
}

template <typename Dtype>
__global__ void forward_projective(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* in, const Dtype* theta, Dtype* source_data, Dtype* out) {//, const Dtype* fill_value_) {

    const int map_size = output_H_ * output_W_;
    
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

        int offset = 8 * n;
		Dtype z = 1 / (x_target * theta[offset + 6] + y_target * theta[offset + 7] + 1);
        Dtype x = x_target * theta[offset] + y_target * theta[offset + 1] + theta[offset + 2];
        Dtype y = x_target * theta[offset + 3] + y_target * theta[offset + 4] + theta[offset + 5];

		/*offset = (n * map_size + h * output_W_ + w) * 3;
		source_data[offset] = (x *= z);
		source_data[offset + 1] = (y *= z);
		source_data[offset + 2] = z;*/

		x = (x * z + (Dtype) 1.) * (width_ - 1) / 2;
        y = (y * z + (Dtype) 1.) * (height_ - 1) / 2;
		offset = (n * map_size + h * output_W_ + w) * 3;
		source_data[offset] = x;
		source_data[offset + 1] = y;
		source_data[offset + 2] = z;

		x = x > 0 ? x : 0; x = x < (width_ - 1) ? x : width_ - 1;
		y = y > 0 ? y : 0; y = y < (height_ - 1) ? y : height_ - 1;
		int w_min = (int)floor(x);
		int w_max = (int)ceil(x);
		int h_min = (int)floor(y);
		int h_max = (int)ceil(y);
		for (int c = 0; c < channels_; ++c) {
			Dtype r = 0;
			offset = (n * channels_ + c) * height_ * width_;
			for (int hh = h_min; hh <= h_max; ++hh) {
				const Dtype dy = (1 - fabs(y - hh));
				for (int ww = w_min; ww <= w_max; ++ww) {
					r += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = r;
		}
        /*int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		for (int c = 0; c < channels_; ++c) {
			Dtype tmp;
			if (h_max < h_min || w_max < w_min) {
				tmp = fill_value_[c];
			}
			else {
				tmp = 0;
				offset = (n * channels_ + c) * height_ * width_;
				for (int hh = h_min; hh <= h_max; ++hh) {
					const Dtype dy = (1 - fabs(y - hh));
					for (int ww = w_min; ww <= w_max; ++ww) {
						tmp += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
					}
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = tmp;
		}*/
    }
}


template <typename Dtype>
__global__ void forward_grid(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* in, const Dtype* theta, Dtype* out) {

    const int map_size = output_H_ * output_W_;
    
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

		int offset = (n * map_size + h * output_W_ + w) * 2;
		Dtype x = theta[offset];
        Dtype y = theta[offset + 1];

        int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		for (int c = 0; c < channels_; ++c) {
			offset = (n * channels_ + c) * height_ * width_;
			Dtype tmp = 0;
			for (int hh = h_min; hh <= h_max; ++hh) {
				const Dtype dy = (1 - fabs(y - hh));
				for (int ww = w_min; ww <= w_max; ++ww) {
					tmp += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = tmp;
		}
    }
}


template <typename Dtype>
__global__ void forward_similarity(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
		const Dtype* in, const Dtype* theta, Dtype* source_data, Dtype* out) {//, const Dtype* fill_value_) {

    const int map_size = output_H_ * output_W_;
    
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

        int offset = 4 * n;
		// 0: alpha
		// 1: scaling
		// 2: tx
		// 3: ty
		Dtype ct = cos(theta[offset]), st = sin(theta[offset]);
		Dtype x = theta[offset + 1] * (x_target * ct - y_target * st) + theta[offset + 2];
		Dtype y = theta[offset + 1] * (x_target * st + y_target * ct) + theta[offset + 3];

		//offset = n * map_size * 2 + h * output_W_ + w;
		offset = (n * map_size + h * output_W_ + w) * 2;
		source_data[offset] = x;
		//source_data[offset + map_size] = y;
		source_data[offset + 1] = y;

        x = (x + (Dtype) 1.) / 2 * (width_ - 1);
        y = (y + (Dtype) 1.) / 2 * (height_ - 1);

        int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		for (int c = 0; c < channels_; ++c) {
			Dtype r = 0;
			offset = (n * channels_ + c) * height_ * width_;
			for (int hh = h_min; hh <= h_max; ++hh) {
				const Dtype dy = (1 - fabs(y - hh));
				for (int ww = w_min; ww <= w_max; ++ww) {
					r += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = r;
		}
    }
}

template <typename Dtype>
__global__ void forward_similarity_plus(const int count, const int channels_,
	const int height_, const int width_, const int output_H_, const int output_W_,
	const Dtype* in, const Dtype* theta, Dtype* source_data, Dtype* out) {//, const Dtype* fill_value_) {

	const int map_size = output_H_ * output_W_;

	CUDA_KERNEL_LOOP(index, count) {
		int n = index / map_size;
		int n_rem = index % map_size;
		int h = n_rem / output_W_;
		int w = n_rem % output_W_;

		Dtype x_target = (Dtype)w / (output_W_ - 1) * 2 - (Dtype)1.;
		Dtype y_target = (Dtype)h / (output_H_ - 1) * 2 - (Dtype)1.;

		int offset = 5 * n;
		// 0: alpha
		// 1: scaling_x
		// 2: scaling_y
		// 3: tx
		// 4: ty
		Dtype ct = cos(theta[offset]), st = sin(theta[offset]);
		Dtype sx = theta[offset + 1], sy = theta[offset + 2];
		Dtype x = sx * x_target * ct - sy * y_target * st + theta[offset + 3];
		Dtype y = sx * x_target * st + sy * y_target * ct + theta[offset + 4];

		x = (x + (Dtype) 1.) / 2 * (width_ - 1);
		y = (y + (Dtype) 1.) / 2 * (height_ - 1);

		//offset = n * map_size * 2 + h * output_W_ + w;
		offset = (n * map_size + h * output_W_ + w) * 2;
		source_data[offset] = x;
		//source_data[offset + map_size] = y;
		source_data[offset + 1] = y;

		int w_min = (floor(x) > 0) ? floor(x) : 0;
		int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
		int h_min = (floor(y) > 0) ? floor(y) : 0;
		int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);

		for (int c = 0; c < channels_; ++c) {
			Dtype r = 0;
			offset = (n * channels_ + c) * height_ * width_;
			for (int hh = h_min; hh <= h_max; ++hh) {
				const Dtype dy = (1 - fabs(y - hh));
				for (int ww = w_min; ww <= w_max; ++ww) {
					r += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = r;
		}
	}
}


template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* theta_data = bottom[1]->gpu_data();
	const int count = num_ * map_size_;

	if (t_type_ == 4) {
		forward_grid<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, theta_data, top_data);

		CUDA_POST_KERNEL_CHECK;
		return;
	}

    Dtype* source_data = source_.mutable_gpu_data();
	
	switch (t_type_) {
	case 0:
		// affine
		forward_affine<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, theta_data, source_data, top_data);//, fill_value_.gpu_data());
		break;
		
	case 1:
		// translation
		forward_translation<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, theta_data, source_data, top_data,
			this->layer_param_.st_param().theta_1_1(), this->layer_param_.st_param().theta_2_2());// , fill_value_.gpu_data());
		break;

	case 2:
		// translation + scaling
		forward_translation_scaling<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, theta_data, source_data, top_data);// , fill_value_.gpu_data());
		break;

	case 3:
		// projective
		forward_projective<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, theta_data, source_data, top_data);// , fill_value_.gpu_data());
		break;

	case 5:
		// similarity
		forward_similarity<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, theta_data, source_data, top_data);//, fill_value_.gpu_data());
		break;

	case 6:
		// similarity+
		forward_similarity_plus<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, theta_data, source_data, top_data);//, fill_value_.gpu_data());
		break;
	}
    
    CUDA_POST_KERNEL_CHECK;
}


///////////////////////////////////////////////////////////////////


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


// compute (1) d{V_i} / d{x_i}, then (2) d{V_i} / d{theta}
// compute sum_{i} d{V_i} / d{U_nm}
template <typename Dtype>
__global__ void backward_affine(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* data, const Dtype* source_data, const Dtype* top_diff,
		Dtype* data_diff, Dtype* theta_diff_cache) {
    
	const int map_size = output_H_ * output_W_;
	
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

        int offset = (n * map_size + h * output_W_ + w) * 2;
        Dtype x = source_data[offset];
        Dtype y = source_data[offset + 1];

		int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		Dtype dv_dx = 0;
        Dtype dv_dy = 0;

		if (data_diff) {
			for (int hh = h_min; hh <= h_max; ++hh) {
				int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
				Dtype dy = 1 - fabs(y - hh);

				for (int ww = w_min; ww <= w_max; ++ww) {
					int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
					Dtype dx = 1 - fabs(x - ww);

					for (int c = 0; c < channels_; ++c) {
						Dtype buffer = top_diff[(n * channels_ + c) * map_size + h * output_W_ + w];
					
						// offset in the input image U
						offset = ((n * channels_ + c) * height_ + hh) * width_ + ww;
						atomic_add(data_diff + offset, buffer * dx * dy);
						
						buffer *= data[offset];
					
						dv_dx += buffer * dy * sign_x;
						dv_dy += buffer * dx * sign_y;
					}
				}
			}
		}else {
			for (int hh = h_min; hh <= h_max; ++hh) {
				int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
				Dtype dy = 1 - fabs(y - hh);
				for (int ww = w_min; ww <= w_max; ++ww) {
					int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
					Dtype dx = 1 - fabs(x - ww);
					for (int c = 0; c < channels_; ++c) {
						Dtype u = 
							top_diff[(n * channels_ + c) * map_size + h * output_W_ + w] *
							data[((n * channels_ + c) * height_ + hh) * width_ + ww];
					
						dv_dx += u * dy * sign_x;
						dv_dy += u * dx * sign_y;
					}
				}
			}
		}
		
		dv_dx *= (Dtype)(width_ - 1) / 2;
		dv_dy *= (Dtype)(height_ - 1) / 2;
		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

		n = n * 6 * map_size + h * output_W_ + w;
		theta_diff_cache[n] = dv_dx * x_target;
		theta_diff_cache[n + map_size] = dv_dx * y_target;
		theta_diff_cache[n + map_size*2] = dv_dx;
		theta_diff_cache[n + map_size*3] = dv_dy * x_target;
		theta_diff_cache[n + map_size*4] = dv_dy * y_target;
		theta_diff_cache[n + map_size*5] = dv_dy;
    }
}

template <typename Dtype>
__global__ void backward_translation(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* data, const Dtype* source_data, const Dtype* top_diff,
		Dtype* data_diff, Dtype* theta_diff_cache) {
    
	const int map_size = output_H_ * output_W_;
	
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

        int offset = (n * map_size + h * output_W_ + w) * 2;
        Dtype x = source_data[offset];
        Dtype y = source_data[offset + 1];

		int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		Dtype dv_dx = 0;
        Dtype dv_dy = 0;

		for (int hh = h_min; hh <= h_max; ++hh) {
			int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
			Dtype dy = 1 - fabs(y - hh);
			for (int ww = w_min; ww <= w_max; ++ww) {
				int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
				Dtype dx = 1 - fabs(x - ww);

				for (int c = 0; c < channels_; ++c) {
					Dtype u = 
						top_diff[(n * channels_ + c) * map_size + h * output_W_ + w] *
						data[((n * channels_ + c) * height_ + hh) * width_ + ww];
					
					dv_dx += u * dy * sign_x;
					dv_dy += u * dx * sign_y;
				}
			}
		}
		
		dv_dx *= (Dtype)(width_ - 1) / 2;
		dv_dy *= (Dtype)(height_ - 1) / 2;

		n = n * 2 * map_size + h * output_W_ + w;
		theta_diff_cache[n] = dv_dx;
		theta_diff_cache[n + map_size] = dv_dy;
    }
}

template <typename Dtype>
__global__ void backward_translation_scaling(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* data, const Dtype* source_data, const Dtype* top_diff,
		Dtype* data_diff, Dtype* theta_diff_cache) {
    
	const int map_size = output_H_ * output_W_;
	
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

        int offset = (n * map_size + h * output_W_ + w) * 2;
        Dtype x = source_data[offset];
        Dtype y = source_data[offset + 1];

		int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		Dtype dv_dx = 0;
        Dtype dv_dy = 0;

		for (int hh = h_min; hh <= h_max; ++hh) {
			int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
			Dtype dy = 1 - fabs(y - hh);
			for (int ww = w_min; ww <= w_max; ++ww) {
				int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
				Dtype dx = 1 - fabs(x - ww);

				for (int c = 0; c < channels_; ++c) {
					Dtype u = 
						top_diff[(n * channels_ + c) * map_size + h * output_W_ + w] *
						data[((n * channels_ + c) * height_ + hh) * width_ + ww];
					
					dv_dx += u * dy * sign_x;
					dv_dy += u * dx * sign_y;
				}
			}
		}
		
		dv_dx *= (Dtype)(width_ - 1) / 2;
		dv_dy *= (Dtype)(height_ - 1) / 2;
		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

		n = n * 4 * map_size + h * output_W_ + w;
		theta_diff_cache[n] = dv_dx * x_target;
		theta_diff_cache[n + map_size] = dv_dx;
		theta_diff_cache[n + map_size*2] = dv_dy * y_target;
		theta_diff_cache[n + map_size*3] = dv_dy;
    }
}

template <typename Dtype>
__global__ void backward_projective(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* data, const Dtype* source_data, const Dtype* top_diff,
		Dtype* data_diff, Dtype* theta_diff_cache) {
    
	const int map_size = output_H_ * output_W_;
	//const Dtype width_const = (Dtype)2 / (Dtype)(width_ - 1);
	//const Dtype height_const = (Dtype)2 / (Dtype)(height_ - 1);
	const Dtype width_const = (Dtype)(width_ - 1) / 2;
	const Dtype height_const = (Dtype)(height_ - 1) / 2;

    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

        int offset = (n * map_size + h * output_W_ + w) * 3;
        Dtype x = source_data[offset];
        Dtype y = source_data[offset + 1];
		Dtype z = source_data[offset + 2];

		//Dtype x = (x0 + (Dtype) 1.) * (width_ - 1) / 2;
		//Dtype y = (y0 + (Dtype) 1.) * (height_ - 1) / 2;
		Dtype x0 = x - width_const, y0 = y - height_const;

		int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		Dtype dv_dx = 0;
        Dtype dv_dy = 0;
		Dtype tmp_source_z = 0;

		for (int hh = h_min; hh <= h_max; ++hh) {
			int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
			Dtype dy = 1 - fabs(y - hh);
			for (int ww = w_min; ww <= w_max; ++ww) {
				int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
				Dtype dx = 1 - fabs(x - ww);

				for (int c = 0; c < channels_; ++c) {
					Dtype u = 
						top_diff[(n * channels_ + c) * map_size + h * output_W_ + w] *
						data[((n * channels_ + c) * height_ + hh) * width_ + ww];
					
					Dtype dv_dx_i = u * dy * sign_x;
					Dtype dv_dy_i = u * dx * sign_y;
					dv_dx += dv_dx_i;
					dv_dy += dv_dy_i;
					tmp_source_z -= dv_dx_i * x0 + dv_dy_i * y0;
				}
			}
		}
		
		dv_dx *= width_const * z;
		dv_dy *= height_const * z;
		tmp_source_z *= z;

		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

		n = n * 8 * map_size + h * output_W_ + w;
		theta_diff_cache[n] = dv_dx * x_target;
		theta_diff_cache[n + map_size] = dv_dx * y_target;
		theta_diff_cache[n + map_size*2] = dv_dx;
		theta_diff_cache[n + map_size*3] = dv_dy * x_target;
		theta_diff_cache[n + map_size*4] = dv_dy * y_target;
		theta_diff_cache[n + map_size*5] = dv_dy;
		theta_diff_cache[n + map_size*6] = tmp_source_z * x_target;
		theta_diff_cache[n + map_size*7] = tmp_source_z * y_target;
    }
}


template <typename Dtype>
__global__ void backward_grid(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* data, const Dtype* theta_data, const Dtype* top_diff,
		Dtype* data_diff, Dtype* theta_diff) {
    
	const int map_size = output_H_ * output_W_;
	
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

        int offset = (n * map_size + h * output_W_ + w) * 2;
        Dtype x = theta_data[offset];
		Dtype y = theta_data[offset + 1];

		int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		Dtype dv_dx = 0;
        Dtype dv_dy = 0;

		for (int hh = h_min; hh <= h_max; ++hh) {
			int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
			Dtype dy = 1 - fabs(y - hh);
			for (int ww = w_min; ww <= w_max; ++ww) {
				int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
				Dtype dx = 1 - fabs(x - ww);

				for (int c = 0; c < channels_; ++c) {
					Dtype u = 
						top_diff[(n * channels_ + c) * map_size + h * output_W_ + w] *
						data[((n * channels_ + c) * height_ + hh) * width_ + ww];
					
					dv_dx += u * dy * sign_x;
					dv_dy += u * dx * sign_y;
				}
			}
		}
		
		theta_diff[offset] = dv_dx;
		theta_diff[offset + 1] = dv_dy;
    }
}


template <typename Dtype>
__global__ void backward_similarity(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
		const Dtype* data, const Dtype* source_data, const Dtype* top_diff, const Dtype* theta_data,
		Dtype* data_diff, Dtype* theta_diff_cache) {
    
	const int map_size = output_H_ * output_W_;
	
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

        int offset = (n * map_size + h * output_W_ + w) * 2;
        Dtype x0 = source_data[offset];
        Dtype y0 = source_data[offset + 1];

		Dtype x = (x0 + (Dtype) 1.) / 2 * (width_ - 1);
		Dtype y = (y0 + (Dtype) 1.) / 2 * (height_ - 1);

		int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		Dtype dv_dx = 0;
        Dtype dv_dy = 0;

		for (int hh = h_min; hh <= h_max; ++hh) {
			int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
			Dtype dy = 1 - fabs(y - hh);
			for (int ww = w_min; ww <= w_max; ++ww) {
				int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
				Dtype dx = 1 - fabs(x - ww);
				for (int c = 0; c < channels_; ++c) {
					Dtype u = 
						top_diff[(n * channels_ + c) * map_size + h * output_W_ + w] *
						data[((n * channels_ + c) * height_ + hh) * width_ + ww];
					
					dv_dx += u * dy * sign_x;
					dv_dy += u * dx * sign_y;
				}
			}
		}
		
		dv_dx *= (Dtype)(width_ - 1) / 2;
		dv_dy *= (Dtype)(height_ - 1) / 2;
		//Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        //Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

		offset = 4 * n;
		n = offset * map_size + h * output_W_ + w;
		
		Dtype s = 1 / theta_data[offset + 1];
		x0 -= theta_data[offset + 2];
		y0 -= theta_data[offset + 3];
		//theta_diff_cache[n] = dv_dx * (ty - y) + dv_dy * (x - tx);	// alpha
		//theta_diff_cache[n + map_size] = dv_dx * 1/s * (tx - x) + dv_dy * 1/s * (y - ty);	// scaling
		theta_diff_cache[n] = dv_dx * (-y0) + dv_dy * (x0);	// alpha
		theta_diff_cache[n + map_size] = s * (dv_dx * (x0) + dv_dy * (y0));	// scaling
		theta_diff_cache[n + map_size * 2] = dv_dx;		// tx
		theta_diff_cache[n + map_size * 3] = dv_dy;		// ty
    }
}



template <typename Dtype>
__global__ void backward_similarity_plus(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
		const Dtype* data, const Dtype* source_data, const Dtype* top_diff, const Dtype* theta_data,
		Dtype* data_diff, Dtype* theta_diff_cache) {
    
	const int map_size = output_H_ * output_W_;
	
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

        int offset = (n * map_size + h * output_W_ + w) * 2;
        Dtype x = source_data[offset];
        Dtype y = source_data[offset + 1];

		int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		Dtype dv_dx = 0;
        Dtype dv_dy = 0;

		for (int hh = h_min; hh <= h_max; ++hh) {
			int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
			Dtype dy = 1 - fabs(y - hh);
			for (int ww = w_min; ww <= w_max; ++ww) {
				int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
				Dtype dx = 1 - fabs(x - ww);
				for (int c = 0; c < channels_; ++c) {
					Dtype u = 
						top_diff[(n * channels_ + c) * map_size + h * output_W_ + w] *
						data[((n * channels_ + c) * height_ + hh) * width_ + ww];
					dv_dx += u * dy * sign_x;
					dv_dy += u * dx * sign_y;
				}
			}
		}
		
		dv_dx *= (Dtype)(width_ - 1) / 2;
		dv_dy *= (Dtype)(height_ - 1) / 2;
		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

		offset = 5 * n;
		n = offset * map_size + h * output_W_ + w;
		
		Dtype ct = cos(theta_data[offset]), st = sin(theta_data[offset]);
		//Dtype sx = 1 / theta_data[offset + 1], sy = 1 / theta_data[offset + 2];
		x -= theta_data[offset + 3];
		y -= theta_data[offset + 4];
		//theta_diff_cache[n] = dv_dx * (ty - y) + dv_dy * (x - tx);	// alpha
		//theta_diff_cache[n + map_size] = dv_dx * 1/s * (tx - x) + dv_dy * 1/s * (y - ty);	// scaling
		theta_diff_cache[n] = dv_dx * (-y) + dv_dy * (x);	// alpha
		theta_diff_cache[n + map_size] = (dv_dx * ct - dv_dy * st) * x_target;	// scaling x
		theta_diff_cache[n + map_size * 2] = (-dv_dx * st + dv_dy * ct) * y_target;	// scaling y
		theta_diff_cache[n + map_size * 3] = dv_dx;		// tx
		theta_diff_cache[n + map_size * 4] = dv_dy;		// ty
    }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();

    Dtype* data_diff = 0;
    Dtype* theta_diff = bottom[1]->mutable_gpu_diff();

	int count = num_ * map_size_;

	if (t_type_ == 4) {
		const Dtype* theta_data = bottom[1]->gpu_data();

		backward_grid<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, channels_,
			height_, width_, output_H_, output_W_,
			bottom_data, theta_data, top_diff,		// input
			data_diff, theta_diff					// output
			);

		CUDA_POST_KERNEL_CHECK;
		return;
	}

    Dtype* theta_diff_cache = theta_diff_cache_.mutable_gpu_data();

    const Dtype* source_data = source_.gpu_data();
    
    if (propagate_down[0]) {
        data_diff = bottom[0]->mutable_gpu_diff();
        caffe_gpu_set<Dtype>(bottom[0]->count(), 0, data_diff);
    }
    //caffe_gpu_set<Dtype>(bottom[1]->count(), 0, theta_diff);  // UNNECCESSARY

	switch (t_type_) {
	case 0:
		// affine
		
		// compute gradient with respect to theta
		backward_affine<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, channels_,
			height_, width_, output_H_, output_W_,
			bottom_data, source_data, top_diff,		// input
			data_diff, theta_diff_cache				// output
			);

		// aggregate gradient for theta 
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 6, 1, map_size_,
			Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		break;
		
	case 1:
		// translation
		backward_translation<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, source_data, top_diff, data_diff, theta_diff_cache);

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 2, 1, map_size_,
			Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		break;

	case 2:
		// translation + scaling
		backward_translation_scaling<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, source_data, top_diff, data_diff, theta_diff_cache);

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 4, 1, map_size_,
			Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		break;

	case 3:
		// projective
		backward_projective<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, source_data, top_diff, data_diff, theta_diff_cache);

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 8, 1, map_size_,
			Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		break;

	case 5:
		// similarity
		backward_similarity<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, source_data, top_diff, bottom[1]->gpu_data(),
			data_diff, theta_diff_cache);

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 4, 1, map_size_,
			Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		break;

	case 6:
		// similarity+
		backward_similarity_plus<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, channels_, height_, width_, output_H_, output_W_,
			bottom_data, source_data, top_diff, bottom[1]->gpu_data(),
			data_diff, theta_diff_cache);

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 4, 1, map_size_,
			Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		break;
	}
	
    CUDA_POST_KERNEL_CHECK;
}
INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);
} // namespace caffe
