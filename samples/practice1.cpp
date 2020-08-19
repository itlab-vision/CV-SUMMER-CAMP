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

	cv:Mat image = cv::imread(imgName);
    // Filter image

int width = parser.get<int>("width");


 
	int height = parser.get<int>("height");


 
	cv::Mat image = cv::imread(imgName);


 



 
    // Filter image
	Filter* gr = new GrayFilter();
	Filter* res = new ResizeFilter(width, height);
    // Show image

	cv::namedWindow("My image", cv::WINDOW_NORMAL);
	//cv::imshow("My image", gr->ProcessImage(image));
	cv::imshow("My image", res->ProcessImage(image));
	cv::waitKey();
    
    
    
    
    return 0;
}
