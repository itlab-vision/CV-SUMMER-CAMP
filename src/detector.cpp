#include "detector.h"
#include <fstream>

DnnDetector::DnnDetector(string _modelPath, string _configPath, string _labelsPath,
	int inputWidth, int inputHeight, Scalar _mean, bool _swapRB, double _scale ) {
	modelPath = _modelPath;
	configPath = _configPath;
	labelsPath = _labelsPath;
	width = inputWidth;
	height = inputHeight;
	mean = _mean;
	swapRB = _swapRB;
	scale = _scale;
	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;

	net = readNet(modelPath, configPath);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);

	//std::string name;
	std::ifstream in(labelsPath);
	numObj = 0;
	in.seekg(0, ios::beg);
	labels.resize(21);

	if (in.is_open())
	{
		while (getline(in, labels[numObj]) && numObj < 20) 
		{
			numObj++;
		}
	}

	in.close();
}

vector<DetectedObject> DnnDetector::Detect(Mat image) {
	Mat inputTensor, tmp;
	vector<DetectedObject> objects;
	int ddepth = CV_32F;
	double thresh = 0.5;
	blobFromImage(image, inputTensor, scale, Size(width, height), mean, swapRB, false, ddepth);
	net.setInput(inputTensor);
	
	tmp = net.forward().reshape(1, 1);
	int rows = tmp.cols / 7;
	int col = tmp.cols;
	tmp = tmp.reshape(1, rows);
	DetectedObject a;
	for (int i = 0; i < rows; i++) {
		double score = tmp.at<float>(i, 2);
		if (tmp.at<float>(i, 2) >= 0.5) {
			cout << " >0.5" << endl;
			a.score = tmp.at<float>(i, 2);
			a.uuid = tmp.at<float>(i, 1);
			a.xLeftBottom = image.cols*tmp.at<float>(i, 3);
			a.yLeftBottom = image.rows*tmp.at<float>(i, 4);
			a.xRightTop = image.cols*tmp.at<float>(i, 5);
			a.yRightTop = image.rows*tmp.at<float>(i, 6);
			
			a.classname = labels[a.uuid];
			objects.push_back(a);
		}
	}

	return objects;
}