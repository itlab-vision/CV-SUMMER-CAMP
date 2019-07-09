#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image)
{
	cvtColor(image, image, COLOR_BGR2GRAY);
	return image;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	this->width = newWidth;
	this->height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image)
{
	Mat temp(width, height, CV_8UC3);
	resize(image, temp, temp.size(), 0, 0, INTER_LINEAR);
	return temp;
}
