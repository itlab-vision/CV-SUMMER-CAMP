#include "detector.h"
DnnDetector::DnnDetector(string _model, string _config, string _label) {
	model = _model;
	config = _config;
	label = _label;
	return;
}
vector<DetectedObject> DnnDetector::Detect(Mat _mat){
	Mat inputTensor;
	Net net = readNet(model, config);
	Mat image = _mat;
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
	
	
	blobFromImage(image, inputTensor, 0.007843, Size(300, 300), { 127.5,127.5,127.5 }, false, false, CV_32F);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);
	prob = prob.reshape(1, prob.cols/7);
	
	vector<DetectedObject> res;
	for (int i = 0; i < prob.rows; i++) {
		DetectedObject obj;
		obj.Left = prob.at<float>(i, 3)*image.cols;
		obj.Bottom = prob.at<float>(i, 4)*image.rows;
		obj.Right = prob.at<float>(i, 5)*image.cols;
		obj.Top = prob.at<float>(i, 6)*image.rows;
		res.push_back(obj);
	}

	return res;
}