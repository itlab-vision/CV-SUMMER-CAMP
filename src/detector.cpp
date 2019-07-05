#include "detector.h"

DnnDetector::DnnDetector(string _model, string _confing, string _labels){
	model	 = _model;
	confing	 = _confing;
	labels	 = _labels;
	return;
};

vector<DetectedObject> DnnDetector::Detect(Mat image) {
	
	Mat inputTensor;
	vector<DetectedObject> res;
	Net net = readNet(model, confing);

	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

	int scale = 0.007843;
	int ddepth = CV_32F;

	blobFromImage(image, inputTensor, 0.007843, Size(500, 500), { 127.5,127.5, 127.5 }, false, false, ddepth);
	net.setInput(inputTensor);

	Mat prob = net.forward();
	prob = prob.reshape(1, 100);
	
	struct DetectedObject detect;
	for (int i = 0; i < 100; i++) {
		res[i].Left = (prob.at<float>(i, 3));
		res[i].Bottom = (prob.at<float>(i, 4));
		res[i].Right = (prob.at<float>(i, 5));
		res[i].Top = (prob.at<float>(i, 6)); 
	}
	return res;
};