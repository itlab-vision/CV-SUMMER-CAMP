#pragma once
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Classificator
{
public:
	virtual Mat Classify(Mat image) = 0 {}
};

class DnnClassificator : Classificator
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
	string classesFile;
	int backendId;
	int targetId;

	Mat PrepareData (Mat image)
	{
		Mat inputTensor;
		blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
		return inputTensor;
	}

public:
	vector<string> classesNames;

	DnnClassificator(
		string model,
		string config,
		int inputWidth,
		int inputHeight,
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

	Mat Classify(Mat image)
	{
		// Convert OpenCV image to input network 4D tensor
		Mat tensor = this->PrepareData(image);
		
		// Set input to network
		net.setInput(tensor);

		// Execute network
		Mat probabilities = net.forward();

		// Convert output 4D tensor to vertor of probabilities
		probabilities.reshape(1, 1);

		// Return vector of probabilities
		return probabilities;
	}

	void loadClassesNames(string filename)
	{
		vector<string> classes;

		{
			ifstream ifs(filename.c_str());
			if (!ifs.is_open())
				cout << "Can't open file with classes names" << endl;
			std::string line;
			while (std::getline(ifs, line))
			{
				classes.push_back(line);
			}
		}
		// TODO: write code here !

		if (classes.size() <= 2)
		{
			cout << "Classes names loaded incorrectly." << endl;
			system("pause");
		}
		else
			cout << "Classes names loaded." << endl;

		classesNames = classes;
	}

	~DnnClassificator()
	{

	}
};