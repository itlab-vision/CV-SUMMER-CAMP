#include "filter.h"

Mat GrayFilter:: ProcessImage(Mat image) {
	Mat resultImage;
	cvtColor(image, resultImage, cv::COLOR_RGB2GRAY);

	return resultImage;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
	width = newWidth;
	height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image) {
	Mat resultImage;
	cv::resize(image, resultImage, cv::Size(width, height), 0, 0);
	return resultImage;
}