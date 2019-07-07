#include "detector.h"
DnnDetector::DnnDetector(string _model, string _config, string _labels, int _inputWidth, int _inputHeight, Scalar _mean,double _scale, bool _swapRB)
{
	model = _model;
	config = _config;
	labels = _labels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	scale = _scale;
	swapRB = _swapRB;
	net = readNet(model, config);

	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;

	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
}

vector <DetectedObject> DnnDetector::Detect(Mat image) 
{
	Mat inputTensor,prob;
	double scale = 0.007843;
	blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
	net.setInput(inputTensor);
	prob = net.forward();
	prob = prob.reshape(1, 1);
	prob = prob.reshape(1, prob.cols / 7);
	vector<DetectedObject> res;
	for (int i = 0; i < prob.rows; i++) {
		DetectedObject obj;
		obj.uuid = (int)round(prob.at<float>(i, 1));
		obj.confidence = prob.at<float>(i, 2);
		obj.Left = prob.at<float>(i, 3)*image.cols;
		obj.Bottom = prob.at<float>(i, 4)*image.rows;
		obj.Right = prob.at<float>(i, 5)*image.cols;
		obj.Top = prob.at<float>(i, 6)*image.rows;
		res.push_back(obj);
	}
	return res;
}