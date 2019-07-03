#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image)
{
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	return gray;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	if ((newWidth >= 0) && (newHeight >= 0))
	{
		width = newWidth;
		height = newHeight;
	}
}

Mat ResizeFilter::ProcessImage(Mat image)
{
	Mat resized;
	resize(image, resized, Size(width, height));
	return resized;
}