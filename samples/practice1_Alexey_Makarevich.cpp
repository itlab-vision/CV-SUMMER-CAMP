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
	Mat Img = imread(imgName);
	

	// Filter image
	string widthS= parser.get<String>("width");
	int width = stoi(widthS);
	string heightS = parser.get<String>("height");
	int height = stoi(heightS);
	ResizeFilter resizeF(width, height);
	Mat resizedImg = resizeF.ProcessImage(Img);
	Mat grayImg = GrayFilter::ProcessImage(Img);

	// Show image
	imshow("Orig", Img);
	imshow("Gray", grayImg);
	imshow("Resized",resizedImg);
	waitKey();




	return 0;
}
