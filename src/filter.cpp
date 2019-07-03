#include "filter.h"



// Gray filter

Mat GrayFilter::ProcessImage(Mat image) {
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);
	return grayImage;

}


// Resize filter

//ResizeFilter::ResizeFilter(int newWidth, int newHeight) : width(newWidth), height(newHeight) {}
ResizeFilter::ResizeFilter(int newWidth, int newHeight) : width(newWidth), height(newHeight) {}

Mat ResizeFilter::ProcessImage(Mat image) {
	Mat resizedImage;
	resize(image, resizedImage, Size(width, height));
	return resizedImage;
}