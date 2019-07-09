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
"{ writer_path                          |        | path to output video				 }"
"{ q ? help usage                       |        | print help message                }";


bool grabFrame(cv::Mat &frame, cv::VideoCapture &cap);
void showOutputVector(const std::vector<DetectedObject> &detObjects);

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

	std::string writerPath(parser.get <std::string>("writer_path"));

	cv::VideoCapture cap;
	cv::VideoWriter writer;
	
	cv::Mat image = cv::imread(imgName, cv::IMREAD_COLOR);
	if (image.empty())
	{
		cap.open(imgName);

		if (cap.isOpened())	cap >> image;
		else
		{
			cap.open(0);
			if (cap.isOpened())	cap >> image;
		}

		if (image.empty())	return -1;
	}

	Detector *detector = new DnnDetector(modelPath, configPath, labelPath, width, height, scale, mean, swapRB);

	while (cv::waitKey(1) != 27)
	{
		if (!grabFrame(image, cap)) break;

		std::int32_t time = clock();
		std::vector<DetectedObject> objects = detector->Detect(image);
		time = clock() - time;

		// Draw
		std::vector<cv::Rect> rectVec(convertToRect(objects));
		for (auto &rect : rectVec)
		{
			cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 4, 8, 0);
		}

		if (cap.isOpened())
		{
			if (!writer.isOpened())
			{
				std::int32_t fps = cap.get(cv::CAP_PROP_FPS);

				bool isColor = false;
				if (image.channels() == 3)	isColor = true;

				writer.open(writerPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, image.size(), isColor);
			}
			if (writer.isOpened())	writer << image;
		}

		std::string text = "Number of Objects: " + std::to_string(rectVec.size());
		cv::putText(image, text, cv::Point(image.cols * 0.05, image.rows * 0.1), 2, 1, cv::Scalar(255, 255, 255), 1, 1, false);

		text = "Time: " + std::to_string(time) + "ms";
		cv::putText(image, text, cv::Point(image.cols * 0.05, image.rows * 0.2), 2, 1, cv::Scalar(255, 255, 255), 1, 1, false);

		cv::namedWindow("output", cv::WINDOW_FREERATIO);
		cv::imshow("output", image);
	}

	cv::destroyAllWindows();

	if (writer.isOpened())	writer.release();

	return 0;
}



// Return image or grab frame from video or frame from default camera
bool grabFrame(cv::Mat &frame, cv::VideoCapture &cap)
{
	if (cap.isOpened())
	{
		cap >> frame;
		if (frame.empty())	return false;
	}
	else if (frame.empty())
	{
		return false;
	}

	return true;
}
void showOutputVector(const std::vector<DetectedObject> &detObjects)
{
	std::uint32_t counter = 0;
	for (auto &obj : detObjects)
	{
		std::cout << "Obj " << counter << ":\t";

		std::cout << "classId: " << obj.classId << ",\t";
		std::cout << "confidence: " << obj.confidence << ",\t";

		std::cout << "L: " << obj.Left << ",\t";
		std::cout << "B: " << obj.Bottom << ",\t";
		std::cout << "R: " << obj.Right << ",\t";
		std::cout << "T: " << obj.Top << std::endl;

		counter++;
	}
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
	return cv::Rect(object.Left, object.Bottom, object.Right - object.Left, object.Top - object.Bottom);
}