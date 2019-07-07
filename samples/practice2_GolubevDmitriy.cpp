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

	int width = parser.get<int>("w");
	int height = parser.get<int>("h");

	std::string modelPath(parser.get <std::string>("model_path"));
	std::string configPath(parser.get <std::string>("config_path"));
	std::string labelPath(parser.get <std::string>("label_path"));

	cv::Scalar mean(parser.get<cv::Scalar>("mean"));
	bool swapRB = parser.get<bool>("swap");

	cv::Mat image = cv::imread(imgName, cv::IMREAD_COLOR);
	if (image.empty())	return -1;


	// ROI
	std::string winname = "ROI";
	cv::Rect rect = cv::selectROI(winname, image);
	cv::destroyWindow(winname);


	//Image classification
	Classificator *classificator = new DnnClassificator(modelPath, configPath, labelPath, width, height, mean, swapRB);

	cv::Mat prob;
	if (!rect.empty())	prob = classificator->Classify(image(rect));
	else				prob = classificator->Classify(image);
	
	if (prob.empty())	return -1;


	//Show result
	((DnnClassificator*)classificator)->showTheBestClasses(5);


	if (classificator)
	{
		delete classificator;
		classificator = nullptr;
	}

	return 0;
}