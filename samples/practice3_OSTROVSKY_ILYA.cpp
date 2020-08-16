#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
"This is an empty application that can be treated as a template for your "
"own doing-something-cool applications.";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  height                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";


void drawRects(Mat& image, const std::vector<DetectedObject>& objects, const std::string& name) {
	for (size_t i = 0; i < objects.size(); ++i) {
		if (objects[i].classname == name) {
			rectangle(image, { objects[i].Left, objects[i].Top }, { objects[i].Right,  objects[i].Bottom }, { 0, 0, 255 });
		}
	}
}


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
	Mat image;

	String imgName;// = parser.get<String>("i");
	String ptm(parser.get<String>("model_path"));
	String ptc(parser.get<String>("config_path"));
	String ptl(parser.get<String>("label_path"));

	uint32_t width = parser.get<int>("width");
	uint32_t height = parser.get<int>("height");

	Scalar mean = parser.get<Scalar>("mean");
	bool swapRB = parser.get<bool>("swap");

	if (imgName.empty()) {
		VideoCapture vc(0);
		Mat temp;
		namedWindow("camera");

		while (1) {
			vc >> temp;
			imshow("camera", temp);

			if (waitKey(1) == 32) {
				image = temp.clone();
				break;
			}
		}
		destroyWindow("camera");
	}
	else {
		image = imread(imgName);
	}

	//Image classification

	DnnDetector detector(ptm, ptc, ptl);
	auto detected = detector.Detect(image, { (int)width, (int)height }, mean, swapRB);
	drawRects(image, detected, "person");

	//Show result
	namedWindow("window");
	imshow("window", image);

	waitKey(0);
	return 0;
}
