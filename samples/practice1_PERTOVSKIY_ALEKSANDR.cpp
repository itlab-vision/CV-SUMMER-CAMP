#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "filter.h"
#include "videostream.h"


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
	int width = std::stoi(parser.get<String>("width"));
	int height = std::stoi(parser.get<String>("height"));
	Mat image = imread(imgName);


    
    // Filter image

	GrayFilter grayFilter;
	Mat grayImage = grayFilter.ProcessImage(image);

	ResizeFilter resizeFilter(width, height);
	Mat resizedImage = resizeFilter.ProcessImage(image);
    // Show image
    
	namedWindow("Gray image", WINDOW_NORMAL);
	imshow("Gray image", grayImage);
	cv::waitKey();

	namedWindow("Resized image", WINDOW_NORMAL);
	imshow("Resized image", resizedImage);
	cv::waitKey();

	
	VideoStream vidStream(0);
	vidStream.streamToWindow();

    
    return 0;
}
