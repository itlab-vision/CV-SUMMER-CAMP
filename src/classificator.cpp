#include "classificator.h"

DnnClassificator::DnnClassificator(string _model, string _config, string _labels, double _inputWidth, double _inputHeight, Scalar _mean, bool _swapRB)
{
	model = _model;
	config = _config;
	labels = _labels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	swapRB = _swapRB;

	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;

	//load net
	net = readNet(model, config);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
}

 Mat DnnClassificator::Classify(Mat image)
{
	 Mat inputTensor;
	 double scale = 0.017;
	 blobFromImage(image, inputTensor, scale, Size(inputWidth,inputHeight), mean, swapRB, false, CV_32F);
	 net.setInput(inputTensor);
	 Mat prob = net.forward();

	 Point classIdPoint;
	 double confidence;
	 minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	 int classId = classIdPoint.x;
	 std::cout << "Class: " << classId << '\n';
	 std::cout << "Confidence: " << confidence << '\n';
	 return prob;
}