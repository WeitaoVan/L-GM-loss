/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_data_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() {
    //if (num_files_ > 1) {
        this->StopInternalThread();
        /*{
            boost::mutex::scoped_lock lock(data_mutex_);
            need_data_ = 2;
            lock.unlock();
            cond_.notify_one();
        }
        thread_->join();*/
    //}
    for (int i = 0; i < hdf_blobs_q[0].size(); ++i) {
        delete hdf_blobs_q[0][i];
    }
    for (int i = 0; i < hdf_blobs_q[1].size(); ++i) {
        delete hdf_blobs_q[1][i];
    }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::InternalThreadEntry() {
  try {
    while (~this->must_stop()) {
    //while (need_data_ != 2) {

        boost::mutex::scoped_lock lock(data_mutex_);
        while (!need_data_) {
            cond_.wait(lock);
        }
        //if (need_data_ == 2) break;
        need_data_ = false;
        //lock.unlock();

        std::vector< Blob<Dtype>* >* other_hdf_blobs;
        if (p_hdf_blobs == &hdf_blobs_q[0])
            other_hdf_blobs = &hdf_blobs_q[1];
        else
            other_hdf_blobs = &hdf_blobs_q[0];

        ++current_file_;
        if (current_file_ == num_files_) {
            current_file_ = 0;
            if (this->layer_param_.hdf5_data_param().shuffle()) {
                std::random_shuffle(file_permutation_.begin(),
                    file_permutation_.end());
            }
            //LOG(INFO) << "Looping around to first file.";
        }

        LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str(), *other_hdf_blobs);

        //continue;
        /*LOG(INFO) << other_hdf_blobs;
        LOG(INFO) << &hdf_blobs_q[1];
        hdf_blobs_q[1].resize(1);
        hdf_blobs_q[1][0] = new Blob<Dtype>();
        vector<int> shape(4);
        shape[0] = 10000;
        shape[1] = 3;
        shape[2] = 256;
        shape[3] = 256;
        (*other_hdf_blobs)[0]->Reshape(shape);
        LOG(INFO) << (*other_hdf_blobs)[0]->mutable_cpu_data() << ' ' << (*other_hdf_blobs)[0]->count();
        memset((*other_hdf_blobs)[0]->mutable_cpu_data(), 0, (size_t)(*other_hdf_blobs)[0]->count() * 4);*/

    }
  }
  catch (boost::thread_interrupted&) {
  //catch (...) {
      // Interrupted exception is expected on shutdown
      LOG(INFO) << "interrupted";
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::InitRand() {
  
    std::srand(unsigned(std::time(0)));

  const HDF5DataParameter &param_ = this->layer_param_.hdf5_data_param();
  const bool needs_rand = param_.mirror() ||
      (this->phase_ == TRAIN && param_.has_crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int HDF5DataLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::LoadHDF5FileData(
    const char* filename, std::vector< Blob<Dtype>* >& hdf_blobs_) {

  LOG(INFO) << "Loading HDF5 file: " << filename << " idx " << current_file_;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size();
  if (hdf_blobs_.empty()) {
      hdf_blobs_.resize(top_size);
      for (int i = 0; i < top_size; ++i) {
          hdf_blobs_[i] = new Blob<Dtype>();
      }
  }

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  for (int i = 0; i < top_size; ++i) {
      //LOG(INFO) << "top: " << i << " " << hdf_blobs_[i]->shape_string();
    //hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    //hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
    //    MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
          MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i]);
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  
  const int num = hdf_blobs_[0]->shape(0);

  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  /*data_permutation_.clear();
  data_permutation_.resize(num);
  for (int i = 0; i < num; i++)
    data_permutation_[i] = i;*/

  const HDF5DataParameter &param_ = this->layer_param_.hdf5_data_param();

  /////////////////////////////////////////////////////
  // Added by Yuanyi Zhong
  //

  const int channels = hdf_blobs_[0]->shape(1);
  int height = hdf_blobs_[0]->shape(2);
  int width = hdf_blobs_[0]->shape(3);
  int image_size = height * width;

  if (param_.has_crop_size() && param_.crop_num() <= 1) {
      int h_off = 0;
      int w_off = 0;
      int crop_size = param_.crop_size();
        // We only do random crop when we do training.
        if (this->phase_ == TRAIN) {
            h_off = Rand(height - crop_size + 1);
            w_off = Rand(width - crop_size + 1);
        }
        else {
            h_off = (height - crop_size) / 2;
            w_off = (width - crop_size) / 2;
        }
        Dtype *data = hdf_blobs_[0]->mutable_cpu_data();
        Dtype *data_out = data;
        for (int i = 0; i < num; ++i)
            for (int c = 0; c < channels; ++c, data += image_size)
                for (int h = 0; h < crop_size; ++h)
                    for (int w = 0; w < crop_size; ++w) {
                        *data_out++ = data[(h + h_off) * width + w + w_off];
                    }
      height = width = crop_size;
      image_size = height * width;
  }

  if (param_.mean_value_size() > 0) {
      CHECK(param_.mean_value_size() == 1 || param_.mean_value_size() == channels) <<
          "Specify either 1 mean_value or as many as channels (first top blob): " << channels;
      vector<Dtype> mean_values_(channels);
      DLOG(INFO) << "subtracting mean values (may take a while):";
      if (param_.mean_value_size() == 1) {
          for (int c = 0; c < channels; ++c) {
              mean_values_[c] = param_.mean_value(0);
              DLOG(INFO) << "mean_value: " << mean_values_[c];
          }
      }
      else {
          for (int c = 0; c < channels; ++c) {
              mean_values_[c] = param_.mean_value(c);
              DLOG(INFO) << "mean_value: " << mean_values_[c];
          }
      }

      Dtype *data = hdf_blobs_[0]->mutable_cpu_data();
      for (int i = 0; i < num; ++i)
          for (int c = 0; c < channels; ++c) {
              caffe_add_scalar(image_size, -mean_values_[c], data);
              data += image_size;
              //for (int k; k < N; ++k)
              //    *data++ -= mean_values_[c];
          }
  }

  if (param_.mirror()) {
      Dtype *data = hdf_blobs_[0]->mutable_cpu_data();
      for (int i = 0; i < num; ++i)
          if (Rand(2)) {
              for (int c = 0; c < channels; ++c, data += image_size)
                  for (int h = 0; h < height; ++h)
                      for (int w = 0; w < width / 2; ++w)
                          //data[h * width + w] = data[h * width + width-1 - w];
                          std::swap(data[h * width + w], data[h * width + width - 1 - w]);
          }
          else {
              data += image_size * channels;
          }
  }
  
  Dtype scale = param_.scale();
  if (scale != 1) {
      //DLOG(INFO) << "Scaled " << hdf_blobs_[0]->count(0) << " numbers by " << scale;
      caffe_scal(num * channels * image_size, scale, hdf_blobs_[0]->mutable_cpu_data());
  }
  
  // Shuffle if needed.
  /*if (param_.shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    LOG(INFO) << "Successully loaded " << num << " rows (shuffled), first:" << data_permutation_[0];
  } else {
    LOG(INFO) << "Successully loaded " << num << " rows";
  }*/
//  LOG(INFO) << "Successully loaded " << num << " rows";
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  HDF5DataParameter &param_ = *this->layer_param_.mutable_hdf5_data_param();
  
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";

  if (this->phase_ == TEST && param_.mirror())
    param_.set_crop_num(param_.crop_num() * 2);
  if (!param_.has_crop_size())
    param_.set_crop_num(0);

  // Read the source to parse the filenames.
  const string& source = param_.source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << source;
  if (num_files_ == 1) {
      hdf_filenames_.push_back(hdf_filenames_[0]);
      //LOG(INFO)
      num_files_++;
  }
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  InitRand();

  // Shuffle if needed.
  if (param_.shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }  
  
  // Load the first HDF5 file and initialize the line counter.
  p_hdf_blobs = &hdf_blobs_q[0];
  LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str(), hdf_blobs_q[0]);
  
  current_row_ = 0;
  
  // Default to identity permutation.
  const int num = (*p_hdf_blobs)[0]->shape(0);
  data_permutation_.clear();
  data_permutation_.resize(num);
  for (int i = 0; i < num; i++) data_permutation_[i] = i;
  if (param_.shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    LOG(INFO) << "Shuffled " << num << " rows, first:" << data_permutation_[0];
  }
  
  need_data_ = true;
  //if (num_files_ > 1) {
      DLOG(INFO) << "Initializing prefetch";
      StartInternalThread();
  //}

  /*LOG(INFO) << &hdf_blobs_q[1];
  hdf_blobs_q[1].resize(1);
  hdf_blobs_q[1][0] = new Blob<Dtype>();
  vector<int> shape(4);
  shape[0] = 10000;
  shape[1] = 3;
  shape[2] = 256;
  shape[3] = 256;
  hdf_blobs_q[1][0]->Reshape(shape);
  //size_t s = (size_t)hdf_blobs_q[1][0]->count() * sizeof(Dtype);
  //Dtype *p = (Dtype*)(Dtype*)malloc(s);
  //LOG(INFO) << s << p;
  //hdf_blobs_q[1][0]->set_cpu_data(p);
  LOG(INFO) << hdf_blobs_q[1][0]->mutable_cpu_data() << ' ' << hdf_blobs_q[1][0]->count();
  memset(hdf_blobs_q[1][0]->mutable_cpu_data(), 0, (size_t)hdf_blobs_q[1][0]->count() * 4);*/
  /*size_t s = 10000LL * 3 * 256 * 256 * 4;
  void *p = malloc(s);
  LOG(INFO) << p << ' ' << s;*/
  //memset(p, 0, s);

  // Reshape blobs.
  const int batch_size = param_.batch_size();
  const int top_size = this->layer_param_.top_size();
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
    top_shape.resize(hdf_blobs_q[0][i]->num_axes());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_q[0][i]->shape(j);
    }
    if (i == 0 && param_.has_crop_size()) {
      // crop_size > 1, multiple crops per input image
      top_shape[0] *= param_.crop_num();
      top_shape[3] = top_shape[2] = param_.crop_size();
    }
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const HDF5DataParameter &param_ = this->layer_param_.hdf5_data_param();
  const int batch_size = param_.batch_size();
  //std::vector< Blob<Dtype>* > &hdf_blobs_ = *p_hdf_blobs;

  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == (*p_hdf_blobs)[0]->shape(0)) {
      //if (num_files_ > 1) {
        /*++current_file_;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          if (this->layer_param_.hdf5_data_param().shuffle()) {
            std::random_shuffle(file_permutation_.begin(),
                                file_permutation_.end());
          }
          DLOG(INFO) << "Looping around to first file.";
        }*/
        //LOG(INFO) << "require next file, cur " << current_file_;
        {
            boost::mutex::scoped_lock lock(data_mutex_);
            
            need_data_ = true;
            if (p_hdf_blobs == &hdf_blobs_q[0])
                p_hdf_blobs = &hdf_blobs_q[1];
            else
                p_hdf_blobs = &hdf_blobs_q[0];

            lock.unlock();
            cond_.notify_one();
        }
      //}
      current_row_ = 0;
      
      // Default to identity permutation.
      const int num = (*p_hdf_blobs)[0]->shape(0);
      data_permutation_.clear();
      data_permutation_.resize(num);
      for (int i = 0; i < num; i++) data_permutation_[i] = i;
      if (param_.shuffle()) {
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
        LOG(INFO) << "Shuffled " << num << " rows, first:" << data_permutation_[0];
      }
    }
    int j = 0;
    if (this->phase_ == TEST && /*param_.has_crop_size() &&*/ param_.crop_num() > 1) {
      const int channels = (*p_hdf_blobs)[0]->shape(1);
      int height = (*p_hdf_blobs)[0]->shape(2);
      int width = (*p_hdf_blobs)[0]->shape(3);
      int image_size = height * width;
      int crop_size = param_.crop_size();
      
      int h_offs[] = {(height - crop_size) / 2, 0, height - crop_size, 0                , height - crop_size};
      int w_offs[] = {(width - crop_size) / 2 , 0, width - crop_size,  width - crop_size, 0                 };
      
      int crop_num = param_.crop_num();
      const Dtype *data = (*p_hdf_blobs)[0]->cpu_data() +
                data_permutation_[current_row_] * channels * image_size;
      Dtype *data_out = top[j]->mutable_cpu_data() + (i*crop_num) * (top[j]->count() / top[j]->shape(0));
      
      if (param_.mirror()) {
          crop_num >>= 1;
          for (int k = 0; k < crop_num; ++k) {
            int h_off = h_offs[k];
            int w_off = w_offs[k];
            for (int c = 0; c < channels; ++c)
                for (int h = 0; h < crop_size; ++h)
                    for (int w = crop_size-1; w >= 0; --w) {
                        *data_out++ = data[c * image_size + (h + h_off) * width + w + w_off];
                    }
          }
      }
      for (int k = 0; k < crop_num; ++k) {
        int h_off = h_offs[k];
        int w_off = w_offs[k];
        for (int c = 0; c < channels; ++c)
            for (int h = 0; h < crop_size; ++h)
                for (int w = 0; w < crop_size; ++w) {
                    *data_out++ = data[c * image_size + (h + h_off) * width + w + w_off];
                }
      }
      
      j++;
    }
    for (; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &(*p_hdf_blobs)[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);
REGISTER_LAYER_CLASS(HDF5Data);

}  // namespace caffe
