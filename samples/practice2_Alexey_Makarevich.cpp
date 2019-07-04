#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/dnn.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  widht                             |        | image width for classification    }"
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
	int widht =parser.get<int>("widht");
	int heigth =parser.get<int>("heigth");
	String model_path(parser.get<String>("model_path"));
	String config_path(parser.get<String>("config_path"));
	String label_path(parser.get<String>("label_path"));
	bool swap = parser.get<bool>("swap");
	Scalar mean = parser.get<Scalar>("mean");

	Mat img = imread(imgName);
	//Image classification
	DnnClassificator classificator(model_path, config_path, label_path, widht, heigth, mean, swap);
	Mat res = classificator.Classify(img);


	//Show result
	Point classIdPoint;
	double confidence;
	minMaxLoc(res.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;

	std::cout << "Class: " << classId << '\n';
	std::cout << "Confidence: " << confidence << '\n';

	return 0;
}



//-i = "C:\CV-SUMMER-CAMP\data\unn_neuromobile.jpg" -w="224" -h"224" -model_path="C:\CV - SUMMER - CAMP\squeezenet\classification\squeezenet\1.1\caffe\squeezenet1.1.caffemodel" -config_path="C:\CV - SUMMER - CAMP\squeezenet\classification\squeezenet\1.1\caffe\squeezenet1.1.prototxt" -label_path="C:\CV - SUMMER - CAMP\data\squeezenet1.1.labels" -mean = "103.94 116.78 123.68" -swap = FALSE