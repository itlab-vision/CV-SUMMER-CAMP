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
	//Process input arguments
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

	String imgName(parser.get<String>("image"));
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");



	Mat Img = imread(imgName);
	imshow("Orig", Img);
	
	Filter* gray = new GrayFilter();
	Mat grayImg = gray->ProcessImage(Img);
	imshow("GrayImage", grayImg);

	Filter* resize = new ResizeFilter(width, height);
	Mat resizedImg = resize->ProcessImage(Img);
	imshow("resizedImage", resizedImg);

	Filter* gauss = new GaussFilter();
	Mat gaussImg = gauss->ProcessImage(Img);
	imshow("gauss filter", gaussImg);

	WebCamVideo video;
	video.getVideo(gray);

	
	waitKey();
	

	return 0;
}
