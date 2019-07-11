#include "detector.h"
DnnDetector::DnnDetector(string modelPath, string configPath, string labelsPath, int width, int height, Scalar mean, bool swapRB) {
	this->modelPath = modelPath;
	this->configPath = configPath;
	this->labelsPath = labelsPath;
	this->width = width;
	this->height = height;
	this->labelsPath = labelsPath;
	this->mean = mean;
	this->swapRB = swapRB;
}

vector<DetectedObject> DnnDetector::Detect(Mat mat) {
	Mat image = mat;
	Net net = readNet(modelPath, configPath);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
	Mat inputTensor;
	DetectedObject str;
	blobFromImage(image, inputTensor, 0.007843, Size(300,300), { 127.5, 127.5, 127.5 }, false, CV_32F);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);
	prob = prob.reshape(1, prob.cols / 7);
	vector<DetectedObject> res;
	for (int i = 0; i < prob.rows; i++) {
		str.uuid = (int)round(prob.at<float>(i, 1));
		str.Left = prob.at<float>(i, 3)*image.cols;
		str.Bottom = prob.at<float>(i, 4)*image.rows;
		str.Right = prob.at<float>(i, 5)*image.cols;
		str.Top = prob.at<float>(i, 6)*image.rows;
	
		res.push_back(str);
	}
	return res;
}