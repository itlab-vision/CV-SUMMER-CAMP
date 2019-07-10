#include <iostream>
#include <fstream>
#include <string>
#include <conio.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  height                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";

int main(int argc, char** argv)
{
	// Process input arguments
	CommandLineParser parser(argc, argv, cmdOptions);
	parser.about(cmdAbout);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	// Load image and init parameters
	vector<string> labels;
	String imgName(parser.get<String>("image"));
	String model_path(parser.get<String>("model_path"));
	String config_path(parser.get<String>("config_path"));
	String labels_path(parser.get<String>("label_path"));
	Scalar scalar(parser.get<Scalar>("mean"));
	bool swap(parser.get<bool>("swap"));
	int width(parser.get<int>("w"));
	int height(parser.get<int>("h"));
	DnnClassificator classificator(model_path, config_path, labels_path, width, height, swap, scalar);
	Mat image = imread(imgName);
	Mat res;

	//Parse labels
	string input;
	ifstream in(labels_path);
	while (getline(in, input))
	{
		labels.push_back(input);
	}

	//Image classification
	Point classIdPoint;
	double confidence;
	int classId;
	minMaxLoc(classificator.Classify(image), nullptr, &confidence, nullptr, &classIdPoint);
	classId = classIdPoint.x;

	//Show result

	cout << classId << " " << labels[classId-1] << endl;
	system("pause");
	return 0;
}
