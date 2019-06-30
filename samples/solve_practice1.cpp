#include <iostream>
#include "filter.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image         | <none> | image to process        }"
"{ w  width         |        | width for image resize  }"
"{ h  height        |        | height for image resize }"
"{ h ? help usage   |        | print help message      }";


int main(int argc, char** argv)
{
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
    string imgName(parser.get<String>("image"));
    Mat source = imread(imgName, 1);

    // Filter image

    GrayFilter grayFilter = GrayFilter();
    ResizeFilter resizeFilter = ResizeFilter(100, 100);
    
    Mat res = grayFilter.ProcessImage(resizeFilter.ProcessImage(source));

    // Show image

    imshow("small grayscale image", res);

    waitKey(0);
    return 0;
}