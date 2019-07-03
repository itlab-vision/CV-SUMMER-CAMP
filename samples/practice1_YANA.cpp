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
	Mat image = imread(imgName);
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");
    
    // Filter image
	Filter* grayFilter = new GrayFilter;
	Mat gray;
	gray = grayFilter->ProcessImage(image);

	Filter* resizeFilter = new ResizeFilter(width, height);
	ResizeFilter b(width, height);
	Mat resize;
	resize = resizeFilter->ProcessImage(image);

    // Show image
	namedWindow("My image", WINDOW_NORMAL);
	imshow("My image", image);
	waitKey();

	namedWindow("Gray image", WINDOW_NORMAL);
	imshow("Gray image", gray);
	waitKey();

	namedWindow("Resize image", WINDOW_NORMAL);
	imshow("Resize image", resize);
	waitKey();
    
    return 0;
}
