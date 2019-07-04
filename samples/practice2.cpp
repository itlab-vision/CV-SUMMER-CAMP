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
"{ m model_path                         |        | path to model                     }"
"{ c config_path                        |        | path to model configuration       }"
"{ l label_path                         |        | path to class labels              }"
"{ s scalar                             |        | vector of mean model values       }"
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
	string imgName(parser.get<string>("image"));
	float width(parser.get<float>("width"));
	float heigth(parser.get<float>("heigth"));
	string model(parser.get<string>("model_path"));
	string config(parser.get<string>("config_path"));
	string label(parser.get<string>("label_path"));
	Scalar mean(parser.get<Scalar>("scalar"));
	bool swap(parser.get<bool>("swap"));


	//Image classification
	imgName = "C:/CV-Intel/CV-SUMMER-CAMP-build/caffe/qwe.jpg"; //C:/CV-Intel/CV-SUMMER-CAMP-build/caffe
	Mat image = imread(imgName);
	//cv::namedWindow("My image", cv::WINDOW_NORMAL);
	//cv::imshow("My image", image);
	//cv::waitKey(0);
	model = "C:/CV-Intel/CV-SUMMER-CAMP-build/caffe/squeezenet1.1.caffemodel";
	config = "C:/CV-Intel/CV-SUMMER-CAMP-build/caffe/squeezenet1.1.prototxt";
	label = "C:/CV-Intel/CV-SUMMER-CAMP-build/caffe/squeezenet1.1.labels";
	double scale = 1;
	width = 632;
	heigth = 475;
	mean = { 1,1,1 };
	swap = false;
	Classificator* dnnClassificator = new DnnClassificator(model, config, label, scale, width, heigth, mean, swap);
	Mat prob = dnnClassificator->Classify(image);


	Point classIdPoint;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;


	
	//Show result

	std::cout << "Class: " << classId << '\n';
	std::cout << "Confidence: " << confidence << '\n';
	cv::waitKey(0);

	return 0;
}
