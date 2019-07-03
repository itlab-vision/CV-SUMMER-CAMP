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
    Mat image = imread(imgName);
    
    //width and height from command line
    int width = parser.get<int>("w");
    int height = parser.get<int>("h");
    
    // Filter image
    GrayFilter to_gray;
    ResizeFilter resize(width, height);
    Mat gray_image = to_gray.ProcessImage(image);
    Mat resized_image = resize.ProcessImage(image);
    Mat resized_gray_image = resize.ProcessImage(gray_image);

    // Show all images: original and modified
    namedWindow("original", WINDOW_NORMAL);
    imshow("original", image);
    namedWindow("gray", WINDOW_NORMAL);
    imshow("gray", gray_image);
    namedWindow("resized", WINDOW_NORMAL);
    imshow("resized", resized_image);
    namedWindow("resized_gray", WINDOW_NORMAL);
    imshow("resized gray", resized_gray_image);
    waitKey();
    return 0;
}
