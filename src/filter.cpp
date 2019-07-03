#include "filter.h"
Mat GrayFilter::ProcessImage(Mat image){
    
    Mat dst(image.size(), CV_8SC3);
    cvtColor(image, dst, COLOR_BGR2GRAY);
    return dst;
    
};

Mat ResizeFilter::ProcessImage(Mat image){
    Mat dst(image.size(), CV_8SC3);
    resize(image, dst, cv::Size(width, height));
    return dst;
};
