#include <iostream>
#include "filter.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image         | <none> | image to process        }"
"{ h ? help usage   |        | print help message      }";


int main(int argc, char** argv)
{
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

	string imgName(parser.get<String>("image"));

	GrayFilter grayFilter = GrayFilter();
	ResizeFilter resizeFilter = ResizeFilter(100, 100);

	Mat source = imread(imgName, 1);
	
	Mat res = grayFilter.ProcessImage(resizeFilter.ProcessImage(source));

	imshow("small grayscale image", res);

	waitKey(0);
	return 0;
}