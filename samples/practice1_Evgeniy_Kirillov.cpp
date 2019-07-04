#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "filter.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image         | <none> | image to process        }"
"{ w  width         | <none> | width for image resize  }"
"{ h  height        | <none> | height for image resize }"
"{ q ? help usage   | <none> | print help message      }";

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

	// Load image
	String imgName(parser.get<String>("image"));
	int imgWidth = stoi(parser.get<String>("width"));
	int imgHeight = stoi(parser.get<String>("height"));
	Mat image = imread(imgName);

	// Filter image
	GrayFilter* gf = new GrayFilter;
	ResizeFilter* rf = new ResizeFilter(imgWidth, imgHeight);
	Mat dst = gf->ProcessImage(image);
	Mat dst2 = rf->ProcessImage(dst);
	// Show image

	imshow("", dst2);
	waitKey();



	return 0;
}
