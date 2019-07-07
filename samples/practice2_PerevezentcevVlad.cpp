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
	int inputWidth = parser.get<int>("width");
	int inputHeight = parser.get<int>("height");
	String caffemodel(parser.get<String>("model_path"));
	String prototxt(parser.get<String>("config_path"));
	String labels(parser.get<String>("label_path"));
	Scalar mean = parser.get<Scalar>("maen");
	bool swapRB = parser.get<bool>("swap");




	//Image classification
	Classificator* classificator = new DnnClassificator(caffemodel, prototxt, labels, inputWidth, inputHeight, mean, swapRB);
	Mat image = imread(imgName);
	Mat output = classificator->Classify(image);
	
	Point classIdPoint;
	double confidence;
	minMaxLoc(output, 0, &confidence, &classIdPoint);
	int classId = classIdPoint.x;

	
	//Show result
	cout << "Class:" << classId << '\n';
	cout << "Confidence:" << confidence << '\n';

	delete classificator;
	

	return 0;
}
