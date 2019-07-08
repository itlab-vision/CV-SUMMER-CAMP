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
bool process(Filter **filter, const cv::Mat &src, cv::Mat &dst);
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

	// Parser command line
	cv::String filepath(parser.get<cv::String>("image"));
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");


	cv::Mat frame;
	cv::VideoCapture cap;

	// Open image, video or default camera
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

	Filter *filter = nullptr;

	std::cout << "Press space to pause" << std::endl;
	std::cout << "Press Esc to exit" << std::endl;
	while (true)
	{
		if (!grabFrame(frame, cap))	break;
		
		// Filter image
		
		// Gray
		cv::Mat gray;
		if (!filter)
		{
			filter = new GrayFilter();
			process(&filter, frame, gray);
		}

		// Resize
		cv::Mat newSizeImage;
		if (!filter)
		{
			filter = new ResizeFilter(width, height);
			process(&filter, frame, newSizeImage);
		}

		// Gaussian
		cv::Mat gaussian;
		if (!filter)
		{
			filter = new GaussianFilter(cv::Size(5, 5));
			process(&filter, frame, gaussian);
		}

		// Barley-Break
		cv::Mat barleyBreak;
		if (!filter)
		{
			filter = new FilterBarleyBreak(11);
			process(&filter, frame, barleyBreak);
		}


		// Show image
		show("Original", frame);
		show("Gray", gray);
		show("Resize", newSizeImage);
		show("Gaussian", gaussian);
		show("Barley-Break", barleyBreak);

		key = cv::waitKey(pause);
		if (key == ' ')	pause *= -1;
		if (key == 27)	break;
	}

	cv::destroyAllWindows();

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


// Filter
bool process(Filter **filter, const cv::Mat &src, cv::Mat &dst)
{
	if (!(*filter))	return false;

	(*filter)->ProcessImage(src, dst);

	if ((*filter))
	{
		delete (*filter);
		(*filter) = nullptr;
	}

	return true;
}



void show(const std::string &winname, const cv::Mat &frame)
{
	if (frame.empty() || winname.empty())	return;

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
	cv::imshow(winname, frame);
	//cv::waitKey();
	//cv::destroyWindow(winname);
}