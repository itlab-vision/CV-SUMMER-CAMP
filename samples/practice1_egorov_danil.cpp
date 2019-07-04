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
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");
	Mat image = imread(imgName);
    
    //Filter image
	GrayFilter gray_filter;
	ResizeFilter resize_filter(width, height);

	Mat src = imread(imgName);
	Mat dst1 = gray_filter.ProcessImage(image);
	Mat dst2 = resize_filter.ProcessImage(image);

    // Show image
	namedWindow("My image", WINDOW_NORMAL);
	imshow("My image", image);
	imshow("grayscale", dst1);
	imshow("resize", dst2);
	waitKey();
    
    
    
    return 0;
}
