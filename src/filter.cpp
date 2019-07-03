#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image) {
    Mat out;
    cv::cvtColor(image, out, COLOR_BGR2GRAY);
    return out;
};

ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;
};

Mat ResizeFilter::ProcessImage(Mat image) {
    Mat out;
    cv::resize(image, out, cv::Size(width, height));
    return out;
}
