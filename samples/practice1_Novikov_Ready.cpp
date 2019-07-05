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
	int a, b;

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
    cv::Mat image = cv::imread("Cat.jpg");
    
    // Color filter image 
	Filter* BW = new GrayFilter();
	//Show color filter image
	cv::namedWindow("This is my image", cv::WINDOW_NORMAL);
	cv::imshow("This is my image", BW->ProcessImage(image));
	cv::waitKey();

	//Size filter image
	cout << "Enter new parametrs\n"<<"width:\n";
	cin >> a;
	cout << "height:\n";
	cin >> b;
	Filter* Inp = new ResizeFilter(a, b);
	//Show size filter image
	cv::imshow("This is my image", Inp->ProcessImage(image));
	cv::waitKey();




    
    
    return 0;
}
