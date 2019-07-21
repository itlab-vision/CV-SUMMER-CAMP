#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image)
{
	Mat tmp;
	cvtColor(image, tmp, COLOR_BGR2GRAY);
	return tmp;
};

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
};

Mat ResizeFilter:: ProcessImage(Mat image)
{
	Mat tmp;
	resize(image, tmp, Size(width, height));
	return tmp;
};