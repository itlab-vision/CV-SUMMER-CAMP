#include "detector.h"
#include <iostream>
#include <fstream>

DnnDetector::DnnDetector(string _path_to_model, string _path_to_confing, string _path_to_labels,
	int _inputWidth, int _inputHeight, Scalar _mean, bool _swapRB)
{
	path_to_model = _path_to_model;
	path_to_confing = _path_to_confing;
	path_to_labels = _path_to_labels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	swapRB = _swapRB;

	net = readNet(path_to_model, path_to_confing);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

	std::ifstream fs(path_to_labels, ios::in | ios::binary);
	Labels.resize(21);
	for (int i = 0; i < 20; i++)
	{
		if (!getline(fs, Labels[i])) break;
	}
}

vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	Mat inputTensor;
	vector<DetectedObject> result;
	blobFromImage(image, inputTensor, 0.007843, Size(inputWidth, inputHeight), mean, swapRB, false);
	net.setInput(inputTensor);
	Mat detections = net.forward();
	detections = detections.reshape(1, 1);
	
	int cols = detections.cols;
	int rows = detections.cols / 7;

	detections = detections.reshape(1, rows);

	cout << detections;

	DetectedObject tmp;
	for (int i = 0; i < rows; i++)
	{
		if (detections.at<float>(i, 2) > 0.9)
		{
			tmp.uuid = detections.at<float>(i, 1);
			tmp.Score = detections.at<float>(i, 2);
			tmp.Left = detections.at<float>(i, 3)*image.cols;
			tmp.Right = detections.at<float>(i, 4)*image.cols;
			tmp.Top = detections.at<float>(i, 5)*image.rows;
			tmp.Bottom = detections.at<float>(i, 6)*image.rows;
			tmp.classname = Labels[tmp.uuid];

			result.push_back(tmp);
		}
	}

	return result;
}


