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
	string imgName(parser.get<string>("image"));
	string model_path(parser.get<string>("model_path"));
	string config_path(parser.get<string>("config_path"));
	string label(parser.get<string>("label"));
	Scalar mean(parser.get<Scalar>("mean"));
	float heigth(parser.get<float>("heigth"));
	float width(parser.get<float>("width"));
	float scale(parser.get<float>("scale"));
	bool swap(parser.get<bool>("swap"));

	Mat image = imread(imgName);
	Classificator* dnnClassificator = new DnnClassificator(model_path, config_path, label, scale, width, heigth, mean, swap);
	Mat prob = dnnClassificator->Classify(image);

	Point classIdPoint;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	
	cout << "Class" << classIdPoint << "\n";
	cout << "Confidence" << confidence << "\n";


	return 0;
}
