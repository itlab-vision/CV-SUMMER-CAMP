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
"{ h  heigth                            |        | image heigth for classification   }"
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

	/*
	int width = parser.get<int>("width");
	int heigth = parser.get<int>("heigth");
	*/

	String model_path(parser.get<String>("model_path"));
	String config_path(parser.get<String>("config_path"));
	String label_path(parser.get<String>("label_path"));
	Scalar mean = parser.get<Scalar>("mean");
	bool swap = parser.get<bool>("swap");

	Mat image = imread(imgName);
	
	
	int width = image.cols;
	int heigth = image.rows;
	

	//Image classification
	DnnClassificator classificator(model_path, config_path, label_path, width, heigth, mean, swap);

	Mat result = classificator.Classify(image);

	Point classIdPoint;
	double probability;
	minMaxLoc(result.reshape(1, 1), 0, &probability, 0, &classIdPoint);
	int classId = classIdPoint.x;
	//Show result

	cout << "rows: " << image.rows<< ", columns: " << image.cols << ", probability: " << probability << ", class: " << classId << endl;

	system("pause");

	return 0;
}