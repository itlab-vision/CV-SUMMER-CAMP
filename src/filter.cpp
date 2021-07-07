#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image) {
	Mat res;
	cvtColor(image, res, COLOR_BGR2GRAY);
	return res;
}
ResizeFilter::ResizeFilter(int newWidth, int newHeight) : width(newWidth), height(newHeight) {}
Mat ResizeFilter::ProcessImage(Mat image) {
	Mat res;
	resize(image, res, Size(width,height));
	return res;
}