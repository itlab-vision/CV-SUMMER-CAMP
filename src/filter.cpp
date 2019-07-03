#include "filter.h"
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

Mat GrayFilter::ProcessImage(Mat image)
{
	Mat image1;
	cvtColor(image, image1, COLOR_BGR2GRAY, 0);
	return image1;
};