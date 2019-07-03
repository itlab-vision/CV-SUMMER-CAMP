#include "filter.h"

Mat GrayFilter::ProcessImage(Mat Image) {
	Mat result;
	cvtColor(Image, result, COLOR_BGR2GRAY);
	return result;
}

Mat ResizeFilter::ProcessImage(Mat Image) {
	Mat result;
	cv::resize(Image, result, Size(width, height));
	return result;
}