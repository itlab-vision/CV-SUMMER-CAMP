#include "filter.h"
#include "filter_SedovaSasha.h"
#include <opencv2/opencv.hpp>

Mat GrayFilter::ProcessImage(Mat image)
{
	//возвращать изображение в оттенках серого
	Mat  gray;
	cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	return gray;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
}
Mat ResizeFilter::ProcessImage(Mat image)
{
	//возвращать изображение нового размера
	Mat NewSize;
	//cvResize(image, NewSize,0);
	//cv::resize(image, NewSize, NewSize.size(), 0, 0, INTER_LINEAR);
	resize(image, NewSize, Size(width,height));
	return NewSize;
}

