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
"{ scale                                |        | scale the image for blob          }"
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

    string model = parser.get<string>("model_path");
    string config = parser.get<string>("config_path");
    string labels = parser.get<string>("label_path");
    int inputHeight = parser.get<int>("w");
    int inputWidth = parser.get<int>("h");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("swap");
    double scale = parser.get<double>("scale");
    
    //Image classification
    Classificator* dnn = new DnnClassificator(model, config, labels,
                        inputWidth, inputHeight, swapRB, scale, mean = mean);
    Mat result = dnn->Classify(image);
    
    Point classIdPoint;
    double confidence;
    minMaxLoc(result, 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    
	//Show result
    std::cout << "Class: " << classId << '\n';
    std::cout << "Confidence: " << confidence << '\n';

	return 0;
}
