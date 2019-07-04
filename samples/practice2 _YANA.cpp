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
	Mat image = imread(imgName);

	String modelPath(parser.get<String>("model_path"));
	String configPath(parser.get<String>("config_path"));
	String labelPath(parser.get<String>("label_path"));
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");
	Scalar mean = parser.get<Scalar>("mean");
	bool swap = parser.get<bool>("swap");

	//Image classification
	Classificator* a = new DnnClassificator(modelPath, configPath, labelPath, width, height, mean, swap);
	Mat prob = a->Classify(image);

	//Show result
	Point classIdPoint;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;

	cout << "Class: " << classId << '\n';
	cout << "Confidence: " << confidence << '\n';

	return 0;
}
