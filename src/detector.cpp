#include "detector.h"
#include <fstream>

ifstream& GotoLine(ifstream& file, int num) 
{
	file.seekg(ios::beg);
	for (int i = 0; i < num - 1; ++i) {
		file.ignore(numeric_limits<streamsize>::max(), '\n');
	}
	return file;
}


DnnDetector::DnnDetector(String path_to_model, String path_to_config, String path_to_lables)
{
	this->path_to_config = path_to_config;
	this->path_to_lables = path_to_lables;
	this->path_to_model = path_to_model;

	net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}

vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	vector<DetectedObject> result;
	Mat inputTensor;
	blobFromImage(image, inputTensor, scale, spatial_size, mean, swapRB);
	net.setInput(inputTensor);
	
		
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);
	int cols = prob.cols / 7;
	prob = prob.reshape(1, cols);

	ifstream labels(path_to_lables);

	for (int i = 0; i < prob.rows; i++)
	{
		string label;
		GotoLine(labels, prob.at<float>(i, 1));
		getline(labels, label);
		result.push_back(DetectedObject(prob.at<float>(i, 1), prob.at<float>(i, 2), prob.at<float>(i, 3), prob.at<float>(i, 4), prob.at<float>(i, 5), prob.at<float>(i, 6),label));
	}
	
	return result;
}

