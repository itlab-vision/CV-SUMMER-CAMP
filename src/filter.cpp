#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image) {
	cvtColor(image, image, COLOR_BGR2GRAY);
	return image;
}


ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
	this->height = newHeight;
	this->width = newWidth;
}


Mat ResizeFilter::ProcessImage(Mat image) {
	resize(image, image, Size(this->width, this->height));  return image;
}