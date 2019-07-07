#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <conio.h>

#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ scale                                |        | scale						     }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";


std::vector<cv::Rect> convertToRect(const std::vector<DetectedObject> &objects);
cv::Rect convertToRect(const DetectedObject &object);

int main(int argc, const char** argv) {
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

	std::int32_t width = parser.get<std::int32_t>("w");
	std::int32_t height = parser.get<std::int32_t>("h");
	
	std::string modelPath(parser.get <std::string>("model_path"));
	std::string configPath(parser.get <std::string>("config_path"));
	std::string labelPath(parser.get <std::string>("label_path"));

	std::double_t scale = parser.get<std::double_t>("scale");
	cv::Scalar mean(parser.get<cv::Scalar>("mean"));
	bool swapRB = parser.get<bool>("swap");

	cv::Mat image = cv::imread(imgName, cv::IMREAD_COLOR);
	if (image.empty())
	{
		cv::VideoCapture cap(0);
		if (cap.isOpened())	cap >> image;

		if (image.empty())	return -1;
	}

	Detector *detector = new DnnDetector(modelPath, configPath, labelPath, width, height, scale, mean, swapRB);
	std::vector<DetectedObject> objects = detector->Detect(image);

	std::vector<cv::Rect> rectVec(convertToRect(objects));

	for (auto &rect : rectVec)
	{
		cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 4, 8, 0);
	}

	cv::namedWindow("output", cv::WINDOW_FREERATIO);
	cv::imshow("output", image);
	cv::waitKey();

	cv::destroyAllWindows();

	return 0;
}



std::vector<cv::Rect> convertToRect(const std::vector<DetectedObject> &objects)
{
	std::vector<cv::Rect> rectVec;

	for (auto &obj : objects)
	{
		rectVec.push_back(convertToRect(obj));
	}

	return rectVec;
}
cv::Rect convertToRect(const DetectedObject &object)
{
	return cv::Rect(object.Left, object.Top, object.Right - object.Left, object.Bottom - object.Top);
}