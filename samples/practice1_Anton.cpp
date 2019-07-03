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
"{ f filter         | <none> | G - gray, R - resize    }"
"{ h  height        | <none> | height for image resize }"
"{ q ? help usage   | <none> | print help message      }";

int main(int argc, char** argv)
{
    // Process input arguments
    CommandLineParser parser(argc, argv, cmdOptions);
    parser.about(cmdAbout);
	string filter;
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
	if (parser.has("filter")) {
		filter = parser.get<string>("filter");
	}
    
    // Load image
    String imgName(parser.get<String>("image"));
	cv::Mat src;
	src = imread(imgName, 1);
	
    
    // Filter image
	if (filter.find('g') != string::npos) {
		GrayFilter f;
		src = f.ProcessImage(src);
	}
	if (filter.find('r') != string::npos) {
		ResizeFilter filt(parser.get<int>("width"), parser.get<int>("height"));
		src = filt.ProcessImage(src);
	}
    // Show image
	cv::imshow("image", src);
	cv::waitKey();
    
    
    
    return 0;
}
