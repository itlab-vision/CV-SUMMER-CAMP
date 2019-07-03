#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image) {
    cv::Mat dst;
    cvtColor(image, dst, COLOR_BGR2GRAY);
    return dst;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image) {
    cv::Size s(width, height);
    Mat dst; 
    resize(image, dst, s);
    return dst;
}