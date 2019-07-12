#include "filter.h"
#include "filter_SedovaSasha.h"
#include <opencv2/opencv.hpp>

Mat GrayFilter::ProcessImage(Mat image)
{
	
	Mat  gray;
	cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	return gray;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
}
Mat ResizeFilter::ProcessImage(Mat image)
{
	
	Mat NewSize;
	resize(image, NewSize, Size(width,height));
	return NewSize;
}

