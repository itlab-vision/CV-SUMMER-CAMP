#include "detector.h"




DnnDetector::DnnDetector(String _modelPath, String _configPath, String _labelsPath, int _inputWidth, int _inputHeight, double _scale, Scalar _mean, bool _swapRB){
	modelPath = _modelPath;
	configPath = _configPath;
	labelsPath = _labelsPath;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	scale = _scale;
	mean = _mean;
	swapRB = _swapRB;

	net = readNet(modelPath, configPath);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}


tbm::TrackedObjects DnnDetector::Detect(Mat image, int frame_idx) {

	int width_source = image.cols; 
	int height_source = image.rows; // Размеры исходного изображения

	Mat resized_frame;
	resize(image, resized_frame, Size(inputWidth, inputHeight));


	Mat inputTensor;
	tbm::TrackedObjects objects;
	blobFromImage(resized_frame, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);


	net.setInput(inputTensor);
	Mat objectsProbability = net.forward().reshape(1,1);
	int objectsNum = objectsProbability.cols / 7;

	for (int i = 0; i < objectsNum; i++) {
		int uuid = objectsProbability.at<float>(0, 1 + i * 7);
		double score = objectsProbability.at<float>(0, 2 + i * 7);
		int left = objectsProbability.at<float>(0, 3 + i * 7) * width_source;
		int bottom = objectsProbability.at<float>(0, 4 + i * 7) * height_source;
		int right = objectsProbability.at<float>(0, 5 + i * 7) * width_source;
		float temp = objectsProbability.at<float>(0, 6 + i * 7);
		int top = objectsProbability.at<float>(0, 6 + i * 7) * height_source;

		Rect rect(left, bottom, (right - left), (top - bottom));
		rect = rect & Rect(Point(), image.size());
		if (rect.empty())
			continue;
		tbm::TrackedObject cur_obj(rect, score, frame_idx, uuid, -1);

		objects.push_back(cur_obj);
	}
	return objects;
}