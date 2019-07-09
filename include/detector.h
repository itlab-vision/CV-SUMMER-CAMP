#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"
#include "tracking_by_matching.hpp"
#include "filter.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace tbm;

class Detector
{
public:
	virtual TrackedObjects Detect(const Mat& image, int frame_id) = 0;
};


class DnnDetector : Detector{
public:
	DnnDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
		int desired_class_id = 1,
		float confidence_threshold = 0.2,
		const String& net_input_name = "data",
		const String& net_output_name = "detection_out",
		double net_scalefactor = 0.007843,
		const Size& net_size = Size(300, 300),
		const Scalar& net_mean = Scalar(127.5, 127.5, 127.5),
		bool net_swapRB = false);

	TrackedObjects Detect(const Mat& image, int frame_id = -1);
private:
	cv::dnn::Net net;
	float confidence_threshold;
	int desired_class_id;
	String net_input_name;
	String net_output_name;
	double net_scalefactor;
	Size net_size;
	Scalar net_mean;
	bool net_swapRB;
};
