#include "detector.h"

DnnDetector::DnnDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
	int desired_class_id,
	float confidence_threshold,
	const String& net_input_name,
	const String& net_output_name,
	double net_scalefactor,
	const Size& net_size,
	const Scalar& net_mean,
	bool net_swapRB)
	:desired_class_id(desired_class_id),
	confidence_threshold(confidence_threshold),
	net_input_name(net_input_name),
	net_output_name(net_output_name),
	net_scalefactor(net_scalefactor),
	net_size(net_size),
	net_mean(net_mean),
	net_swapRB(net_swapRB)
{
	net = dnn::readNet(net_caffe_model_path, net_caffe_weights_path);
	if (net.empty())
		CV_Error(Error::StsError, "Cannot read Caffe net");
}


TrackedObjects DnnDetector::Detect(const Mat & image, int frame_id) {
	ResizeFilter resized(net_size.width, net_size.height);
	Mat blob = blobFromImage(resized.ProcessImage(image), net_scalefactor, net_size, net_mean, net_swapRB);

	net.setInput(blob, net_input_name);
	Mat forward = net.forward(net_output_name);
	Mat detected(forward.size[2], forward.size[3], CV_32F, forward.ptr<float>());

	TrackedObjects result;
	for (uint32_t i = 0; i < detected.rows; ++i) {
		cv::tbm::TrackedObject obj;

		obj.frame_idx = frame_id;
		obj.confidence = detected.at<float>(i, 2);
		obj.object_id = static_cast<int>(detected.at<float>(i, 1));
		if (desired_class_id >= 0 && desired_class_id != obj.object_id) {
			continue;
		}
		if (obj.confidence < confidence_threshold) {
			continue;
		}

		int x_left = static_cast<int>(detected.at<float>(i, 3) * image.cols);
		int y_bottom = static_cast<int>(detected.at<float>(i, 4) * image.rows);
		int x_right = static_cast<int>(detected.at<float>(i, 5) * image.cols);
		int y_top = static_cast<int>(detected.at<float>(i, 6) * image.rows);

		obj.rect = Rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom)) & Rect(Point(), image.size());
		if (!obj.rect.empty()) {
			result.push_back(obj);
		}
	}

	return result;
}
