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
	int width(parser.get<int>("width"));
	int height(parser.get<int>("heigth"));




	string model = "C:\\Users\\temp2019\\GitProject\\classification\\squeezenet\\1.1\\caffe\\squeezenet1.1.caffemodel";
	string config = "C:\\Users\temp2019\\GitProject\\classification\squeezenet\\1.1\\caffe\\squeezenet1.1.prototxt";
	string labels = "C:\\Users\temp2019\GitProject\classification\squeezenet\1.1\caffe\squeezenet1.1.labels";
	Scalar sc(0, 0, 0, 0);

	Mat src = imread(imgName);
	DnnClassificator dnnClassificator(model,config,labels,width,height,sc,false);
	
	//Image classification
	
	src=dnnClassificator.Classify(src);
	//Show result

	waitKey();
	return 0;
}
