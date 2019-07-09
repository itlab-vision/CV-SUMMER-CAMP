#include "filter.h"

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	this->width = newWidth;
	this->height = newHeight;
}

Mat ResizeFilter::ProcessImage(const Mat& image) {
	Mat result;
	cv::resize(image, result, { width, height });
	return result;;
}
