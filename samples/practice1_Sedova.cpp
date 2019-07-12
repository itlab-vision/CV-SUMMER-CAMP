#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "filter_SedovaSasha.h"

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
	String path_image = parser.get<String>("i");
	Mat image = cv::imread(path_image);
	GrayFilter g1;
	ResizeFilter r1(200,200);
    // Filter image
	Mat GImage = g1.ProcessImage(image);
	Mat RImage = r1.ProcessImage(image);
    // Show image
	imshow("Gray", GImage);
	imshow("Image", image);
	imshow("Resize", RImage);
	waitKey();
    
    return 0;
}
