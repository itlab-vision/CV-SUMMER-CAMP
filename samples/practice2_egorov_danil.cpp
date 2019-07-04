#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
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
	String imgName(parser.get<String>("image"));


	//Image classification
	String model_path(parser.get<String>("model_path"));
	String config_path(parser.get<String>("config_path"));
	String label_path(parser.get<String>("label_path"));
	int width(parser.get<int>("width"));
	int heigth(parser.get<int>("heigth"));
	bool swap(parser.get<bool>("swap"));
	Scalar mean(parser.get<Scalar>("mean"));

	Mat image = imread(imgName);
	DnnClassificator Classificator(model_path, config_path, label_path, width, heigth, mean, swap);
	Mat prob = Classificator.Classify(image);

	//Show result
	Point classIdPoint;
	double confidence;
	minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;

	std::cout << "Class: " << classId << '\n';
	std::cout << "Confidence: " << confidence << '\n';

	return 0;
}
