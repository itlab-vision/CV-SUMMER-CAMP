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


	//Mat image = imread(string(argv[1]));
    
    // Filter image


    // Show image
	imshow("My precious kartinka", image);
	waitKey();
	destroyWindow("My precious kartinka");

	Filter* filter = new GrayFilter;
	image = filter->ProcessImage(image);

	imshow("My precious gray kartinka", image);
	waitKey();
	destroyWindow("My precious gray kartinka");

	cout << "Enter preferable weight & height int integers, please :)" << endl;
	int width, height;
	cin >> width;
	cin >> height;	

	filter = new ResizeFilter(width, height);
	image = filter->ProcessImage(image);

	imshow("My precious resized kartinka", image);
	waitKey();
	destroyWindow("My precious resized kartinka");
    
    
    
    return 0;
}
