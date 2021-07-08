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
	int width(parser.get<int>("width"));
	int height(parser.get<int>("height"));

    String imgName(parser.get<String>("image"));
	Mat image = imread(imgName);

	if (image.empty()) {
		std::cout << "Error: the image has been incorrectly loaded." << std::endl;
		return 0;
	}
	if (width == 0 || height == 0) {
		std::cout << "Error: width or height are zero." << std::endl;
		return 0;
	}

    // Filter image
	GrayFilter *grayFilter = new GrayFilter();
	ResizeFilter *resizeFilter= new ResizeFilter(width, height);

	Mat grayImage = grayFilter->ProcessImage(image);
	Mat resizeImage = resizeFilter->ProcessImage(grayImage);

    // Show image
	imshow("display", resizeImage);
	waitKey(0);

    return 0;
}
