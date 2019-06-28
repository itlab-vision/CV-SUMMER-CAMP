#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
	virtual vector<DetectedObject> Detect(Mat image) = 0 {}
	virtual ~Detector() = 0 {}
};

class DnnDetector : Detector
{
private:
	Net net;
	float scale;
	Scalar mean;
	bool swapRB;
	int inputWidth;
	int inputHeight;
	string framework;
	string model;
	string config;
	string classesNames;
	int backendId;
	int targetId;

	Mat PrepareData (Mat image)
	{
		Mat inputTensor;
		blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
		return inputTensor;
	}

public:

	DnnDetector(
		string model,
		string config,
		int inputWidth,
		int inputHeight,
		float scale = 1.0f,
		Scalar mean = Scalar(0.0, 0.0, 0.0, 0.0),
		bool swapRB = false,
		int backendId = 0,
		int targetId = 0)
	{
		// Load net
		net = readNet(model, config, framework);
		net.setPreferableBackend(backendId);
		net.setPreferableTarget(targetId);
		
		// Copy parameters
		this->inputWidth = inputWidth;
		this->inputHeight = inputHeight;
		this->scale = scale;
		this->mean = mean;
		this->swapRB = swapRB;
		this->backendId = backendId;
		this->targetId = targetId;

		if (net.empty())
		{
			std::cout << "Can't load network by using the following files:" << endl;
			std::cout << "config:   " << config << endl;
			std::cout << "model: " << model << endl;
			system("pause");
		}
		else
			cout << "Neural Network loaded." << endl;
	}

	vector<DetectedObject> Detect(Mat image)
	{
		// Convert OpenCV image to input network 4D tensor
		Mat tensor = this->PrepareData(image);
		
		// Set input to network
		net.setInput(tensor);

		// Execute network
		Mat result = net.forward();

		// Convert output 4D tensor to vertor of probabilities
		result = result.reshape(1, 1);
		result = result.reshape(1, 200);
		vector<DetectedObject> shapes = vector<DetectedObject>();

		for (int i = 0; i < result.rows; i++)
		{
			DetectedObject shape;
			shape.xLeft =(int)(image.rows * result.at<float>(i, 3));
			shape.yBottom =((int)(image.cols * result.at<float>(i, 4)));
			shape.xRight =((int)(image.rows * result.at<float>(i, 5)));
			shape.yTop =((int)(image.cols * result.at<float>(i, 6)));

			shapes.push_back(shape);
		}

		// Return vector shapes
		return shapes;
	}

	~DnnDetector()
	{

	}
};