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
<<<<<<< HEAD
	//Process input arguments
=======
	// Process input arguments
>>>>>>> f1832c4255139dda91caea09b0405b018360b3a2
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

<<<<<<< HEAD
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
	
=======
	// Load image
	String imgName(parser.get<String>("image"));
	Mat Img = imread(imgName);
	

	// Filter image
	string widthS= parser.get<String>("width");
	int width = stoi(widthS);
	string heightS = parser.get<String>("height");
	int height = stoi(heightS);
	ResizeFilter resizeF(width, height);
	Mat resizedImg = resizeF.ProcessImage(Img);
	Mat grayImg = GrayFilter::ProcessImage(Img);
	//Mat GaussImg = GaussFilter::ProcessImage(Img); ne rabotaet T_T

	// Show image
	imshow("Orig", Img);
	//imshow("Gauss", GaussImg);
	imshow("Gray", grayImg);
	imshow("Resized",resizedImg);
	waitKey();



>>>>>>> f1832c4255139dda91caea09b0405b018360b3a2

	return 0;
}
