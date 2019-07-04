#include "filter.h"

Mat GrayFilter::ProcessImage(Mat Image) {
	Mat dst;
	cv::cvtColor(Image, dst, COLOR_BGR2GRAY, 0);
	return dst;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
	width = newWidth;
	height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat Image) {
	Size s(width, height);
	Mat dst;
	resize(Image, dst, s);
	return dst;
}