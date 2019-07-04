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
	string model= parser.get<string>("model_path");
	string config= parser.get<string>("config_path");
	string label= parser.get<string>("label_path");
	int width(parser.get<int>("width"));
	int height(parser.get<int>("heigth"));
	Scalar scalar = parser.get<Scalar>("mean");
	bool swapRB = parser.get<int>("swap");
	

	Mat src = imread(imgName);
	DnnClassificator dnnClassificator(model,config,label,width,height,scalar,swapRB);
	
	//Image classification
	Point classIdPoint;
	double confidence;
	Mat dst =dnnClassificator.Classify(src);
	minMaxLoc(dst, 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	//Show result

	cout << "Class:" << classId << endl;
	cout << "Confidence:" << confidence << endl;
	return 0;
}
