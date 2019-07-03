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
	Size size(1, 1);
	Mat dst;
	//cv::resize(image, dst,size,0,0,interpolation);
	cv::resize(image, dst, Size(), 0, 0.1, INTER_LINEAR);
	return dst;
}