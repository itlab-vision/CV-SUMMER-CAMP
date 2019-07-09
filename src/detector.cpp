#include "detector.h"

DnnDetector::DnnDetector(string _model, string _confing, string _labels){
	model	 = _model;
	confing	 = _confing;
	labels	 = _labels;
	

	net = readNet(model, confing);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
};

vector<DetectedObject> DnnDetector::Detect(Mat image) {
	
	Mat inputTensor;
	vector<DetectedObject> res;
	

	

	int scale = 0.007843;
	int ddepth = CV_32F;

	blobFromImage(image, inputTensor, 0.007843, Size(300, 300), { 127.5,127.5, 127.5 }, false, false, ddepth);
	net.setInput(inputTensor);

	Mat prob = net.forward();
	prob = prob.reshape(1, 1);

	int rows = prob.cols / 7;
	prob = prob.reshape(1, rows);
	
	for (int i = 0; i < rows; i++) {
		DetectedObject detect;

		detect.Left = (prob.at<float>(i, 3));
		detect.Bottom = (prob.at<float>(i, 4));
		detect.Right = (prob.at<float>(i, 5));
		detect.Top = (prob.at<float>(i, 6));

		res.push_back(detect);
	}
	return res;
};