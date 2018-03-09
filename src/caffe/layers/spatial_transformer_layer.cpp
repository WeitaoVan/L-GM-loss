#include <cmath>
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/spatial_transformer_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
	const string &transform_type = this->layer_param_.st_param().transform_type();
	if (transform_type == "affine") {
		t_type_ = 0;
	}
	else if (transform_type == "translation") {
		t_type_ = 1;
	}
	else if (transform_type == "translation_scaling") {
		t_type_ = 2;
	}
	else if (transform_type == "projective") {
		t_type_ = 3;
	}
	else if (transform_type == "grid") {
		t_type_ = 4;
	}
    else if (transform_type == "similarity") {
		t_type_ = 5;
	}
	else if (transform_type == "similarity+") {
		t_type_ = 6;
	}
	else {
		LOG(FATAL) << "Transformation type: " << transform_type << " not supported!";
	}
    output_H_ = bottom[0]->shape(2);
    if (this->layer_param_.st_param().has_output_h()) {
        output_H_ = this->layer_param_.st_param().output_h();
    }
    output_W_ = bottom[0]->shape(3);
    if (this->layer_param_.st_param().has_output_w()) {
        output_W_ = this->layer_param_.st_param().output_w();
    }
    map_size_ = output_H_ * output_W_;
    LOG(INFO) << "Spatial Transformer: output (" << output_H_ << ',' << output_W_
		<< "), type: " << transform_type;
    
	if (Caffe::mode() == Caffe::CPU) {
		// target coordinates (grid), [x_t, y_t, 1]
		target_.Reshape(1, 3, output_H_, output_W_);
		Dtype* target_data = target_.mutable_cpu_data();
		// -1 <= x_t,y_t <= 1
		for (int h = 0; h < output_H_; ++h)
			for (int w = 0; w < output_W_; ++w) {
				// for x
				target_data[target_.offset(0, 0, h, w)] = (Dtype)w / (output_W_ - 1) * 2 - (Dtype)1.;
				// for y
				target_data[target_.offset(0, 1, h, w)] = (Dtype)h / (output_H_ - 1) * 2 - (Dtype)1.;
				// for constant 1
				target_data[target_.offset(0, 2, h, w)] = (Dtype)1.0;
			}
	}
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->shape()[0];
    channels_ = bottom[0]->shape()[1];
    height_ = bottom[0]->shape()[2];
    width_ = bottom[0]->shape()[3];
    
	switch (t_type_) {
	case 0:
		CHECK_EQ(bottom[1]->shape(1), 6) << "Second blob should be 6-dimension theta for affine";
		// create source gradient cache for different channels
		if (Caffe::mode() == Caffe::GPU)
			theta_diff_cache_.Reshape(num_, 6, output_H_, output_W_);
		// create source coordinates, [x_s,y_s] when multiplied by a [2x3] matrix
		source_.Reshape(num_, 2, output_H_, output_W_);
		break;
	case 1:
		CHECK_EQ(bottom[1]->shape(1), 2) << "Second blob should be 2-dimension theta for translation";
		if (Caffe::mode() == Caffe::GPU)
			theta_diff_cache_.Reshape(num_, 2, output_H_, output_W_);
		source_.Reshape(num_, 2, output_H_, output_W_);
		break;
	case 2:
		CHECK_EQ(bottom[1]->shape(1), 4) << "Second blob should be 4-dimension theta for translation+scaling";
		if (Caffe::mode() == Caffe::GPU)
			theta_diff_cache_.Reshape(num_, 4, output_H_, output_W_);
		source_.Reshape(num_, 2, output_H_, output_W_);
		break;
	case 3:
		CHECK_EQ(bottom[1]->shape(1), 8) << "Second blob should be 8-dimension theta for projective";
		if (Caffe::mode() == Caffe::GPU)
			theta_diff_cache_.Reshape(num_, 8, output_H_, output_W_);
		source_.Reshape(num_, 3, output_H_, output_W_);
		break;
	case 4:
		CHECK_EQ(bottom[1]->count(1), map_size_*2) << "Second blob should be twice the size of output blob for grid";
		//source_.Reshape(num_, 2, output_H_, output_W_);
		break;
    case 5:
		CHECK_EQ(bottom[1]->shape(1), 4) << "Second blob should be 4-dimension theta for similarity";
		if (Caffe::mode() == Caffe::GPU)
			theta_diff_cache_.Reshape(num_, 4, output_H_, output_W_);
		source_.Reshape(num_, 2, output_H_, output_W_);
		break;
	case 6:
		CHECK_EQ(bottom[1]->shape(1), 5) << "Second blob should be 5-dimension theta for similarity+";
		if (Caffe::mode() == Caffe::GPU)
			theta_diff_cache_.Reshape(num_, 5, output_H_, output_W_);
		source_.Reshape(num_, 2, output_H_, output_W_);
		break;
	}

    top[0]->Reshape(num_, channels_, output_H_, output_W_);
    
	if (Caffe::mode() == Caffe::GPU) {
		vector<int> all_ones_shape;
		all_ones_shape.push_back(map_size_);
		theta_diff_op_.Reshape(all_ones_shape);
		caffe_set<Dtype>(map_size_, 1, theta_diff_op_.mutable_cpu_data());
	}

	/*int s = this->layer_param_.st_param().fill_value_size();
	vector<int> sh; sh.push_back(channels_);
	fill_value_.Reshape(sh);
	if (s == 0) {
		caffe_set(channels_, (Dtype)0, fill_value_.mutable_cpu_data());
	}
	else {
		CHECK(s == channels_) << "fill_value length should equal to num of channels";
		for (int i = 0; i < s; ++i) {
			fill_value_.mutable_cpu_data()[i] = this->layer_param_.st_param().fill_value(i);
		}
	}*/
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* theta_data = bottom[1]->cpu_data();
    
	// grid sampler
	if (t_type_ == 4) {
		for (int n = 0; n < num_; ++n) {
			// sample U -> V given source coordinate. O(W*H). Bilinear sampling.
			for (int h = 0; h < output_H_; ++h) {
				for (int w = 0; w < output_W_; ++w) {
					// get the mapped coordinate [x_s,y_s] on U
					int offset = (h * output_W_ + w) << 1;
					Dtype x = theta_data[offset];
					Dtype y = theta_data[offset + 1];

					int w_min = std::max(int(floor(x)), 0);
					int w_max = std::min(int(ceil(x)), width_ - 1);
					int h_min = std::max(int(floor(y)), 0);
					int h_max = std::min(int(ceil(y)), height_ - 1);
					for (int c = 0; c < channels_; ++c) {
						Dtype r = 0;
						for (int hh = h_min; hh <= h_max; ++hh) {
							const Dtype dy = (1 - fabs(y - hh));
							for (int ww = w_min; ww <= w_max; ++ww) {
								r += bottom[0]->data_at(n, c, hh, ww)*(1 - fabs(x - ww)) * dy;
							}
						}
						top_data[top[0]->offset(n, c, h, w)] = r;
					}
				}
			}
		}
		return;
	}

	const Dtype* target_data = target_.cpu_data();
    Dtype* source_data = source_.mutable_cpu_data();
    //caffe_set<Dtype>(top[0]->count(), 0, top_data);
    
	for (int n = 0; n < num_; ++n) {
		Dtype *src_data = source_data + n * 2 * map_size_;

		switch (t_type_) {
		case 0:
			// affine	
			// compute source coordinate, source = [2x3][3*map_size_] = theta * target
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, map_size_, 3, Dtype(1.0),
				theta_data + n * 6, target_data, Dtype(0.), src_data);
			// compute source in real source coordinate range
			//   x: [-1,1] -> [0,width-1]
			//   y: [-1,1] -> [0,height-1]
			caffe_add_scalar(2 * map_size_, (Dtype)1., src_data);
			break;

		case 1:
			// translation
			//caffe_copy(map_size_ * 2, target_data, src_data);
			caffe_cpu_axpby(map_size_, (Dtype)this->layer_param_.st_param().theta_1_1(), target_data, (Dtype)0, src_data);
			caffe_cpu_axpby(map_size_, (Dtype)this->layer_param_.st_param().theta_2_2(), target_data + map_size_, (Dtype)0, src_data + map_size_);
			caffe_add_scalar(map_size_, (Dtype)1. + theta_data[n * 2], src_data);
			caffe_add_scalar(map_size_, (Dtype)1. + theta_data[n * 2 + 1], src_data + map_size_);
			break;

		case 2:
			// translation + scaling
			caffe_cpu_axpby(map_size_, theta_data[n * 4], target_data, (Dtype)0, src_data);
			caffe_cpu_axpby(map_size_, theta_data[n * 4 + 2], target_data + map_size_, (Dtype)0, src_data + map_size_);
			caffe_add_scalar(map_size_, (Dtype)1. + theta_data[n * 4 + 1], src_data);
			caffe_add_scalar(map_size_, (Dtype)1. + theta_data[n * 4 + 3], src_data + map_size_);
			break;

		case 3:
			// projective
			{
				// compute source coordinate, source = [3x3][3*map_size_] = theta * target
				Dtype t[9];
				caffe_copy(8, theta_data + n * 8, t);
				t[8] = 1;
				src_data = source_data + n * 3 * map_size_;
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 3, map_size_, 3, Dtype(1.0),
					t, target_data, Dtype(0.), src_data);
				caffe_div(map_size_, src_data, src_data + 2 * map_size_, src_data);
				caffe_div(map_size_, src_data + map_size_, src_data + 2 * map_size_, src_data + map_size_);

				// compute source in real source coordinate range
				//   x: [-1,1] -> [0,width-1]
				//   y: [-1,1] -> [0,height-1]
				caffe_add_scalar(2 * map_size_, (Dtype)1., src_data);
			}
			break;
		}

		caffe_scal<Dtype>(map_size_, (Dtype)(width_ - 1) / (Dtype) 2., src_data);
		caffe_scal<Dtype>(map_size_, (Dtype)(height_ - 1) / (Dtype) 2., src_data + map_size_);

		// sample U -> V given source coordinate. O(W*H). Bilinear sampling.
		for (int h = 0; h < output_H_; ++h) {
			for (int w = 0; w < output_W_; ++w) {
				// get the mapped coordinate [x_s,y_s] on U
				int offset = h * output_W_ + w;
				Dtype x = src_data[offset]; //source_data[source_.offset(n, 0, h, w)];
				Dtype y = src_data[offset + map_size_]; //source_data[source_.offset(n, 1, h, w)];

				x = std::min(std::max(x, (Dtype)0), (Dtype)(width_ - 1));
				y = std::min(std::max(y, (Dtype)0), (Dtype)(height_ - 1));
				int w_min = (int)floor(x);
				int w_max = (int)ceil(x);
				int h_min = (int)floor(y);
				int h_max = (int)ceil(y);
				for (int c = 0; c < channels_; ++c) {
					Dtype r = 0;
					for (int hh = h_min; hh <= h_max; ++hh) {
						const Dtype dy = (1 - fabs(y - hh));
						for (int ww = w_min; ww <= w_max; ++ww) {
							r += bottom[0]->data_at(n, c, hh, ww)*(1 - fabs(x - ww)) * dy;
						}
					}
					top_data[top[0]->offset(n, c, h, w)] = r;
				}
				/*int w_min = std::max(int(floor(x)), 0);
				int w_max = std::min(int(ceil(x)), width_ - 1);
				int h_min = std::max(int(floor(y)), 0);
				int h_max = std::min(int(ceil(y)), height_ - 1);
				//*** TODO: 'for channel' should be move outside h,w-loop for performance!!
				// But no performance loss for gray scale image :)
				for (int c = 0; c < channels_; ++c) {
					Dtype r;
					if (h_max < h_min || w_max < w_min) {
						r = fill_value_.cpu_data()[c];
					}
					else {
						r = 0;
						for (int hh = h_min; hh <= h_max; ++hh) {
							const Dtype dy = (1 - fabs(y - hh));
							for (int ww = w_min; ww <= w_max; ++ww) {
								r += bottom[0]->data_at(n, c, hh, ww)*(1 - fabs(x - ww)) * dy;
							}
						}
						top_data[top[0]->offset(n, c, h, w)] = r;
					}
				}*/
			}
		}
	}
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
	
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* data_diff = 0;
    Dtype* theta_diff = bottom[1]->mutable_cpu_diff();
    
	if (t_type_ == 4) {
		const Dtype* theta_data = bottom[1]->cpu_data();

		for (int n = 0; n < num_; ++n) {
			for (int h = 0; h < output_H_; ++h) {
				for (int w = 0; w < output_W_; ++w) {
					int offset = (h * output_W_ + w) << 1;
					Dtype x = theta_data[offset];
					Dtype y = theta_data[offset + 1];
					int w_min = std::max(int(floor(x)), 0);
					int w_max = std::min(int(ceil(x)), width_ - 1);
					int h_min = std::max(int(floor(y)), 0);
					int h_max = std::min(int(ceil(y)), height_ - 1);
					Dtype tmp_source_x = 0;
					Dtype tmp_source_y = 0;

					for (int hh = h_min; hh <= h_max; ++hh) {
						int sign_y = hh >= y ? 1 : -1;
						Dtype dy = 1 - fabs(y - hh);
						for (int ww = w_min; ww <= w_max; ++ww) {
							int sign_x = ww >= x ? 1 : -1;
							Dtype dx = 1 - fabs(x - ww);
							for (int c = 0; c < channels_; ++c) {
								Dtype u = top_diff[top[0]->offset(n, c, h, w)] *
									bottom[0]->data_at(n, c, hh, ww);
								tmp_source_x += u*dy*sign_x;
								tmp_source_y += u*dx*sign_y;
							}
						}
					}
					theta_diff[offset] = tmp_source_x;
					theta_diff[offset + 1] = tmp_source_y;
				}
			}
		}
		return;
	}
		
	
	const Dtype* target_data = target_.cpu_data();
    const Dtype* source_data = source_.cpu_data();
    Dtype* source_diff = source_.mutable_cpu_diff();

    if (propagate_down[0]) {
        data_diff = bottom[0]->mutable_cpu_diff();
        caffe_set<Dtype>(bottom[0]->count(), 0, data_diff);
//        caffe_set<Dtype>(source_.count(), 0, source_diff);
    }
	
	const Dtype width_const = (Dtype)(width_ - 1) / (Dtype)2.;
	const Dtype height_const = (Dtype)(height_ - 1) / (Dtype)2.;

    for (int n = 0; n < num_; ++n) {
        if (propagate_down[0]) {
            for (int h = 0; h < output_H_; ++h) {
                for (int w = 0; w < output_W_; ++w) {
                    Dtype x = source_data[source_.offset(n, 0, h, w)];
                    Dtype y = source_data[source_.offset(n, 1, h, w)];
					int w_min = std::max(int(floor(x)), 0);
					int w_max = std::min(int(ceil(x)), width_ - 1);
					int h_min = std::max(int(floor(y)), 0);
					int h_max = std::min(int(ceil(y)), height_ - 1);
					Dtype tmp_source_x = 0;
                    Dtype tmp_source_y = 0;
                  
                    for (int hh = h_min; hh <= h_max; ++hh) {
						int sign_y = hh >= y ? 1 : -1; //caffe_sign<Dtype>(hh - y);//(y <= (Dtype)hh ) ? 1 : -1;
						Dtype dy = 1 - fabs(y - hh);
                        for (int ww = w_min; ww <= w_max; ++ww) {
                            // *** slightly different from original paper, at point ww == x.
							//  In the original paper, the derivative is defined as 1 when ww == x, whereas it's 0 here.
                            int sign_x = ww >= x ? 1 : -1; // caffe_sign<Dtype>(ww - x);
                            Dtype dx = 1 - fabs(x - ww);
                            for (int c = 0; c < channels_; ++c) {
                                // d(L)/d(U^{c}_{nm})=\sum_{j} d(L)/d(V^{c}_{j}) * d(V^{c}_{j})/d(U^{c}_{nm})
                                // bottom_diff[(n,c,hh,ww)]=\sum_{j} top_diff[(n,c,h,w)] * eq(6) (an error)
                                Dtype buffer = top_diff[top[0]->offset(n, c, h, w)];
                                data_diff[bottom[0]->offset(n, c, hh, ww)] += buffer * dx * dy;
                                // d(L)/d(x_{j})=\sum_{c} d(L)/d(V^{c}_j)*d(V^{c}_j)/d(x_{j})
                                // source_diff[(n,0,h,w)] = \sum_{c} top[(n,c,h,w)] * \sum_{nm} U_{nm} max
                                buffer *= bottom[0]->data_at(n,c,hh,ww);
                                tmp_source_x += buffer*dy*sign_x;
                                tmp_source_y += buffer*dx*sign_y;
                            }
                        }
                    }
					source_diff[source_.offset(n,0,h,w)] = tmp_source_x * width_const;
					source_diff[source_.offset(n,1,h,w)] = tmp_source_y * height_const;
                }
            }
        }else {
            for (int h = 0; h < output_H_; ++h) {
                for (int w = 0; w < output_W_; ++w) {
                    Dtype x = source_data[source_.offset(n, 0, h, w)];
                    Dtype y = source_data[source_.offset(n, 1, h, w)];
					int w_min = std::max(int(floor(x)), 0);
					int w_max = std::min(int(ceil(x)), width_ - 1);
					int h_min = std::max(int(floor(y)), 0);
					int h_max = std::min(int(ceil(y)), height_ - 1);
                    Dtype tmp_source_x = 0;
                    Dtype tmp_source_y = 0;
                  
                    for (int hh = h_min; hh <= h_max; ++hh) {
						int sign_y = hh >= y ? 1 : -1; //caffe_sign<Dtype>(hh - y);
						Dtype dy = 1 - fabs(y - hh);
                        for (int ww = w_min; ww <= w_max; ++ww) {
							int sign_x = ww >= x ? 1 : -1; //caffe_sign<Dtype>(ww - x);
							Dtype dx = 1 - fabs(x - ww);
                            for (int c = 0; c < channels_; ++c) {
                                Dtype buffer = top_diff[top[0]->offset(n, c, h, w)];
                                buffer *= bottom[0]->data_at(n,c,hh,ww);
                                tmp_source_x += buffer*dy*sign_x;
                                tmp_source_y += buffer*dx*sign_y;
                            }
                        }
                    }
                    source_diff[source_.offset(n,0,h,w)] = tmp_source_x * width_const;
                    source_diff[source_.offset(n,1,h,w)] = tmp_source_y * height_const;
                }
            }
        }
        // d(L)/d(theta)
		switch (t_type_) {
		case 0:
			// affine
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2, 3, map_size_,
				(Dtype)1., source_diff + n * 2 * map_size_, target_data, (Dtype)0., theta_diff + n * 6);
			break;
		case 1:
			// translation
			theta_diff[n * 2] = caffe_cpu_dot(map_size_, source_diff + n * 2 * map_size_, target_data + 2 * map_size_);
			theta_diff[n * 2 + 1] = caffe_cpu_dot(map_size_, source_diff + n * 2 * map_size_ + map_size_, target_data + 2 * map_size_);
			break;
		case 2:
			// translation+scaling
			theta_diff[n * 4] = caffe_cpu_dot(map_size_, source_diff + n * 2 * map_size_, target_data);
			theta_diff[n * 4 + 1] = caffe_cpu_dot(map_size_, source_diff + n * 2 * map_size_, target_data + 2 * map_size_);
			theta_diff[n * 4 + 2] = caffe_cpu_dot(map_size_, source_diff + n * 2 * map_size_ + map_size_, target_data + map_size_);
			theta_diff[n * 4 + 3] = caffe_cpu_dot(map_size_, source_diff + n * 2 * map_size_ + map_size_, target_data + 2 * map_size_);
			break;
		case 3:
			//LOG(FATAL) << "no cpu implementation";

			break;
		}
    }
}


#ifdef CPU_ONLY
STUB_GPU(SpatialTransformerLayer);
#endif

INSTANTIATE_CLASS(SpatialTransformerLayer);
REGISTER_LAYER_CLASS(SpatialTransformer);

} // namespace caffe
