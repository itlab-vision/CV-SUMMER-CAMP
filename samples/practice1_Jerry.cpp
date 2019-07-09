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

    // Filter image
	Filter* filter = new GrayFilter;
	Mat resultGray = filter->ProcessImage(image);

	int width(parser.get<int>("width"));
	int height(parser.get<int>("height"));
	filter = new ResizeFilter(width, height);
	Mat resultResize = filter->ProcessImage(image);

    // Show image
	imshow("My precious kartinka", image);
	waitKey();
	destroyWindow("My precious kartinka");	

	imshow("My precious gray kartinka", resultGray);
	waitKey();
	destroyWindow("My precious gray kartinka");	

	imshow("My precious resized kartinka", resultResize);
	waitKey();
	destroyWindow("My precious resized kartinka"); 
    
    return 0;
}
