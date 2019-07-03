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
"{ f filter         | <none> | choose your filter(0 - resize , 1 - )}"
"{ q ? help usage   | <none> | print help message      }";

int main(int argc, char** argv)
{
    // Process input arguments
    CommandLineParser parser(argc, argv, cmdOptions);
    parser.about(cmdAbout);

    int filter = 0;
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (parser.has("filter")) {
        filter = parser.get<int>("filter");
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    // Load image
    String imgName(parser.get<String>("image"));
    GrayFilter g;
    ResizeFilter r(parser.get<int>("width"), parser.get<int>("height"));
    cv::Mat src, dst;
    src = imread(imgName);
    // Filter image

    switch (filter)
    {    
        case 0:
            dst = r.ProcessImage(src);
        break;
        default:
            dst = g.ProcessImage(src);
        break;
    }
    // Show image
    cv::imshow("image", src);
    cv::imshow("image2", dst);
    waitKey(0);

    return 0;
}
