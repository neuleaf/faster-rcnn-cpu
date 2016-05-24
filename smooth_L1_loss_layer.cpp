// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
  sigma2_ = loss_param.sigma() * loss_param.sigma();
  has_weights_ = (bottom.size() >= 3);
  if (has_weights_) {
    CHECK_EQ(bottom.size(), 4) << "If weights are used, must specify both "
      "inside and outside weights";
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
    CHECK_EQ(bottom[0]->channels(), bottom[3]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[3]->height());
    CHECK_EQ(bottom[0]->width(), bottom[3]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  // vector of ones used to sum
  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  for (int i = 0; i < bottom[0]->count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //NOT_IMPLEMENTED;
  
  // cpu implementation
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  int count = bottom[0]->count();
  caffe_sub(count, 
  			bottom[0]->cpu_data(), 
  			bottom[1]->cpu_data(),
  			diff_.mutable_cpu_data());
  
  if(has_weights_){
  	caffe_mul(count, 
  			  bottom[2]->cpu_data(), 
  			  diff_.cpu_data(), 
  			  diff_.mutable_cpu_data());
  }
  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
  //        |x| - 0.5 / sigma / sigma    otherwise
  const Dtype* in = diff_.cpu_data();
  Dtype* out = errors_.mutable_cpu_data();
  for(int index=0; index<count; ++index){
  	Dtype val = in[index];
  	Dtype abs_val = abs(val);
  	if(abs_val < 1.0 / sigma2_){
  		out[index] = 0.5 * val * val * sigma2_;
  	}
  	else{
  		out[index] = abs_val - 0.5 / sigma2_;
  	}
  }
  
  if(has_weights_){
  	caffe_mul(count, bottom[3]->cpu_data(), out, errors_.mutable_cpu_data());
  }
  
  // compute loss
  Dtype loss = caffe_cpu_dot(count, ones_.cpu_data(), errors_.cpu_data());
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
  // end cpu implementation
  
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //NOT_IMPLEMENTED;
  // cpu implementation
  int count = diff_.count();
  const Dtype* in = diff_.cpu_data();
  Dtype* out = diff_.mutable_cpu_data();
  for(int index=0; index < count; index++){
  	Dtype val = in[index];
  	Dtype abs_val = abs(val);
  	if(abs_val < 1.0 / sigma2_){
  		out[index] = sigma2_ *  val;
  	} 
  	else{
  		out[index] = (Dtype(0) < val) - (val < Dtype(0));
  	}
  }
  
  for(int i=0; i<2; ++i){
  	if(propagate_down[i]){
  		const Dtype sign = (i == 0) ? 1 : -1;
  		const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
  		caffe_cpu_axpby(
  			count, 
  			alpha, 
  			out,//diff_.cpu_data(), 
  			Dtype(0), 
  			bottom[i]->mutable_cpu_diff());
  		
  		if(has_weights_){
  			caffe_mul(
  				count, 
  				bottom[2]->cpu_data(), 
  				bottom[i]->cpu_diff(), 
  				bottom[i]->mutable_cpu_data());
  			caffe_mul(
  				count,
  				bottom[3]->cpu_data(),
  				bottom[i]->cpu_diff(),
  				bottom[i]->mutable_cpu_data());
  		}
  	}
  }
  // end cpu implementation
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
