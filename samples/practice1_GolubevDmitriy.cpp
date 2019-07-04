#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "filter.h"

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image         | <none> | image to process        }"
"{ w  width         | <none> | width for image resize  }"
"{ h  height        | <none> | height for image resize }"
"{ q ? help usage   | <none> | print help message      }";

bool grabFrame(cv::Mat &frame, cv::VideoCapture &cap);
void show(const std::string &winname, const cv::Mat &frame);

int main(int argc, char** argv)
{
	// Process input arguments
	cv::CommandLineParser parser(argc, argv, cmdOptions);
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

	// Parser
	cv::String filepath(parser.get<cv::String>("image"));
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");


	cv::Mat frame;
	cv::VideoCapture cap;
	frame = cv::imread(filepath, cv::IMREAD_COLOR);
	if (frame.empty())
	{
		cap.open(filepath);
		if (!cap.isOpened())
		{
			cap.open(0);
			if (!cap.isOpened())	return -1;
		}
	}


	std::int8_t pause = 1;
	std::uint8_t key = NULL;

	while (true)
	{
		if (!grabFrame(frame, cap))	break;

		// Filter image

		// Gray
		cv::Mat gray;
		GrayFilter gf;
		gf.ProcessImage(frame, gray);

		// Resize
		cv::Mat newSizeImage;
		ResizeFilter rf(width, height);
		rf.ProcessImage(frame, newSizeImage);

		// Gaussian
		cv::Mat gaussian;
		GaussianFilter gausF(cv::Size(11, 11));
		gausF.ProcessImage(frame, gaussian);


		// Show image

		// Original
		show("Original", frame);

		// Gray
		show("Gray", gray);

		// Resize
		show("Resize", newSizeImage);

		// Gaussian
		show("Gaussian", gaussian);

		key = cv::waitKey(pause);
		if (key == ' ')	pause *= -1;
		if (key == 27)	break;
	}

	cv::destroyAllWindows();

	return 0;
}



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



void show(const std::string &winname, const cv::Mat &frame)
{
	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
	cv::imshow(winname, frame);
	//cv::waitKey();
	//cv::destroyWindow(winname);
}