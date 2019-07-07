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
	int height(parser.get<int>("height"));
	int width(parser.get<int>("width"));
	String model_path(parser.get<String>("model_path"));
	String config_path(parser.get<String>("config_path"));
	String label_path(parser.get<String>("label_path"));
	Scalar mean(parser.get<Scalar>("mean"));
	bool swapRB(parser.get<bool>("swap"));
	Mat image = imread(imgName);
	/*String imgName = "C:\\CV-SUMMER-CAMP\\data\\lobachevsky.jpg";
	Mat image = imread(imgName);
	int height = 227;
	int width = 227;
	resize(image, image, Size(width, height));
	String model_path = "C:\\openvino\\models\\classification\\squeezenet\\1.1\\caffe\\squeezenet1.1.caffemodel";
	String config_path = "C:\\openvino\\models\\classification\\squeezenet\\1.1\\caffe\\squeezenet1.1.prototxt";
	String label_path = "C:\\CV-SUMMER-CAMP\\data\\squeezenet1.1.labels";
	Scalar mean = (0, 0, 0);
	bool swapRB = false;*/
	//Image classification
	Classificator* clr = new DnnClassificator(model_path, config_path, label_path, width, height, swapRB, mean);
	Mat output = clr->Classify(image);
	Point classIdPoint;
	double confidence;
	minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;

	//Show result
	cout << "The image is " << clr->DecodeLabel(classId) << " with confidence " << confidence * 100 << "%";
	delete clr;
	return 0;
}
