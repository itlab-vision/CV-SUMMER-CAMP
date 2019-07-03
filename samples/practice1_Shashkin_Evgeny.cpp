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
		int imgWidth(parser.get<int>("width"));
		int imgHeight(parser.get<int>("height"));

		Mat image = imread(imgName);
    
    // Filter image
		GrayFilter gray;
		Mat grayImage;
		grayImage=gray.ProcessImage(image);

		ResizeFilter resize(imgWidth,imgHeight);
		Mat resizedImage;
		resizedImage = resize.ProcessImage(image);

    // Show image
		namedWindow("Gray image", WINDOW_AUTOSIZE);
		imshow("Gray image", grayImage);
		waitKey(0);

		namedWindow("Resized image", WINDOW_AUTOSIZE);
		imshow("Resized image", resizedImage);
		waitKey(0);
    
    return 0;
}
