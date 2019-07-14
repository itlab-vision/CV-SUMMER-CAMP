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
	String path_image = parser.get<String>("i");
	string path_to_model = parser.get<String>("model_path");
	string path_to_config=parser.get<String>("config_path");
	string path_to_labesparser=parser.get<String>("label_path");
	int width = 10;
		//=parser.get<int>("w");
	int height = 10;
		//= parser.get<int>("h");
	Scalar mean= Scalar(0, 0, 0, 0);
	int swapRB=0;
	Mat image = cv::imread(path_image);
	DnnClassificator dnn_class(path_to_model, path_to_config, path_to_labesparser, width, height, mean, swapRB);
	Mat dnnImage = dnn_class.Classify(image);
	//Image classification
	Point classIdPoint;
	double confidence;
	dnn_class.Classify(image);
	minMaxLoc(dnnImage.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	//Show result
	imshow("image", image);
	cout << "class:" << classId << "\n";
	cout << "confidence:" << confidence<<"";
	waitKey();
	return 0;
}
