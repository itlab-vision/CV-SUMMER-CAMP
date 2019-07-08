#include "detector.h"

DnnDetector::DnnDetector(string path_to_model, string path_to_config, string path_to_labels, int _inputWidth, int _inputHeight, bool _swapRB, Scalar _mean)
{
	model_path = path_to_model;
	config_path = path_to_config;
	labels_path = path_to_labels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	swapRB = _swapRB;
	mean = _mean;
	net = readNet(model_path, config_path);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}

vector <DetectedObject> DnnDetector::Detect(Mat image)
{
	Mat inputTensor;
	blobFromImage(image, inputTensor, 0.007843, Size(inputWidth, inputHeight), mean, swapRB, false);
	net;
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);

	int rows = prob.cols / 7;
	prob = prob.reshape(1, rows);

	
	vector <DetectedObject> res;

	for (int i = 0; i < rows; i++)
	{
		DetectedObject tmp;
		tmp.uuid = prob.at<float>(i, 1);
		tmp.Left = prob.at<float>(i, 3)*image.cols;
		tmp.Bottom = prob.at<float>(i, 4)*image.rows;
		tmp.Right = prob.at<float>(i, 5)*image.cols;
		tmp.Top = prob.at<float>(i, 6)*image.rows;
		res.push_back(tmp);
	}
	
	return res;
}