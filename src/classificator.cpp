#include "classificator.h"
#include <fstream>

DnnClassificator::DnnClassificator(string _modelPath, string _configPath, string _labelsPath,
	int inputWidth, int inputHeight, Scalar _mean, bool _swapRB) {
	
	modelPath = _modelPath;
	configPath = _configPath;
	labelsPath = _labelsPath;
	width = inputWidth;
	height = inputHeight;
	mean = _mean;
	swapRB = _swapRB;
	classesNames.resize(1000);
	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;
	std::string name;
	std::ifstream in(labelsPath); 
	int count = 0;
    if (in.is_open())
	{
			while (getline(in, classesNames[count]))
			{
				if (count == 999)
				{
					break;
				}
				count++;

			}
	}
	net = readNet(modelPath);//, configPath);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
}

Mat DnnClassificator::Classify(Mat image) {
	Mat inputTensor;
	double scale = 1.0 / 127.50000414375013;
	int ddepth = CV_32F;
	blobFromImage(image, inputTensor, scale, Size(width, height), mean, swapRB, false, ddepth);
	net.setInput(inputTensor);
	

	return net.forward();
}