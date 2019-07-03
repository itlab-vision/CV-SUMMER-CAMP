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

    int width(parser.get<int>("w"));
    int height(parser.get<int>("h"));

    // Load image
    String imgName(parser.get<String>("image"));
    Mat img = cv::imread(imgName);
    cv::imshow("Original image", img);
    cv::waitKey();

    // Filter image - grayscale
    GrayFilter grayFilter = GrayFilter();
    img = grayFilter.ProcessImage(img);
    cv::imshow("Grayscale image", img);
    cv::waitKey();

    // Filter image - resize
    ResizeFilter rzFilter = ResizeFilter(width, height);
    img = rzFilter.ProcessImage(img);
    cv::imshow("Resized image", img);
    cv::waitKey();

    return 0;
}
