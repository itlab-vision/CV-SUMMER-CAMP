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
	String model_path(parser.get<String>("model_path"));
	String config_path(parser.get<String>("config_path"));
	String label_path(parser.get<String>("label_path"));
	Point classIdPoint;
	double confidence;
	Mat image = imread(imgName);

	//Image classification
	DnnClassificator *classOfImage = new DnnClassificator(model_path, config_path, label_path, image.cols, image.rows, false);
	Mat result = classOfImage->Classify(image);
	cout << imgName << endl << model_path << endl << config_path << endl << label_path << endl;
	//Show result
	minMaxLoc(result, 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;

	cout << "Class: " << classId << endl;
	cout << "Confidence: " << confidence << endl;
	imshow("", result);
	waitKey();

	return 0;
}
