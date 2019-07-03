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
	Mat image;
	
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
	int width(parser.get<int>("w"));
	int height(parser.get<int>("h"));
	GrayFilter gFilter;
	ResizeFilter rFilter(width, height);
	image = imread(imgName);
	
	

    
    // Filter image
	image = gFilter.ProcessImage(image);
	image = rFilter.ProcessImage(image);
	
	
	
	
    // Show image
	imshow("New window", image);
	waitKey(0);
    
    
    
    return 0;
}
