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
    String ImgName(parser.get<String>("image"));
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");
    Mat src = imread(ImgName);
    
    // Filter image
	
	Filter *GR_image = new GrayFilter();
	Filter *RSZ_image = new ResizeFilter(width, height);

    // Show image

    namedWindow("My_image", WINDOW_NORMAL);
    imshow("My_image", src);
    waitKey();
    
	namedWindow("Gray image", WINDOW_NORMAL);
	imshow("Gray image", GR_image->ProcessImage(src));
	waitKey();
    
	namedWindow("New size image", WINDOW_NORMAL);
	imshow("New size image", RSZ_image->ProcessImage(src));
	waitKey();

    
    return 0;
}
