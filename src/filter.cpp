#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image)
{
	cvtColor(image, image, COLOR_BGR2GRAY);
	return image;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image)
{
	resize(image, image, cv::Size(width, height));
	return image;
}
