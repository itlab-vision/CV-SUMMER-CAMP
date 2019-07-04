#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image) 
{
	Mat dst;
	cv::cvtColor(image, dst, COLOR_BGR2GRAY, 0);
	return dst;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight) 
{
	width = newWidth;
	height = newHeight;
}
Mat ResizeFilter::ProcessImage(Mat image) 
{
	Mat dst;
	cv::resize(image, dst, Size(width,height));
	return dst;
}