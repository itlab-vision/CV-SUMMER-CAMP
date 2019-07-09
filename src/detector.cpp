#include "detector.h"

DnnDetector::DnnDetector(string path_to_model, string path_to_config, string path_to_labels) 
{
	this->path_to_model = path_to_model;
	this->path_to_config = path_to_config;
	this->path_to_labels = path_to_labels;

	Net net = readNet(this->path_to_model, this->path_to_config);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	this->net = net;
}


vector<DetectedObject> DnnDetector::Detect(Mat image) 
{
	Mat inputTensor;
	blobFromImage(image, inputTensor, scale, Size(input_width, input_height), mean, swapRB, false);
	net.setInput(inputTensor);
	Mat output = net.forward().reshape(1, 1);
	output = output.reshape(1, output.cols / 7);

	vector<DetectedObject> resObj;

	for (int i = 0; i < output.rows; i++)
	{
		DetectedObject res;
		res.classid = output.at<float>(i, 1);
		res.className = this->ParseLabels((int)output.at<float>(i, 1) - 1);
		res.confidence = output.at<float>(i, 3);
		res.Left = output.at<float>(i, 3) * image.cols;
		res.Bottom = output.at<float>(i, 4) * image.rows;
		res.Right = output.at<float>(i, 5) * image.cols;
		res.Top = output.at<float>(i, 6) * image.rows;

		resObj.push_back(res);
	}
	

	return resObj;
}


string DnnDetector::ParseLabels(int class_id) 
{
	ifstream in(path_to_labels);
	vector<string> classesNames;
	string line;
	while (getline(in, line)) {
		classesNames.push_back(line);
	}
	return classesNames[class_id];
}