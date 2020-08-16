#include "filter.h"



Mat GrayFilter::ProcessImage(Mat image)
{
	Mat result;
	cvtColor(image, result, COLOR_BGR2GRAY);
	return result;
}


ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
}


Mat ResizeFilter::ProcessImage(Mat image)
{
	Mat result;
	resize(image, result, { width, height }, 0, 0);
	return result;
}
