#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image)
{
	Mat dst(image.size(), CV_8UC1);
	cvtColor(image, dst, COLOR_BGR2GRAY);
	return dst;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
	: width(newWidth), height(newHeight) {}

Mat ResizeFilter::ProcessImage(Mat image)
{
	Mat dst(image.size(), image.type());
	resize(image, dst, Size(width, height));
	return dst;
}
