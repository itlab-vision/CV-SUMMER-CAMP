#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image) {
    Mat grayed;
    cvtColor(image, grayed, COLOR_BGR2GRAY);
    return grayed;

}

ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image) {
    cv::Size new_size(width, height);
    Mat changed;
    resize(image, changed, new_size);
    return changed;
}
