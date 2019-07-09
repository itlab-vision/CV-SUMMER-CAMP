#include "filter.h"
Mat GrayFilter::ProcessImage(Mat image){
    
    Mat dst;
    cvtColor(image, dst, COLOR_BGR2GRAY);
    return dst;
    
};

Mat ResizeFilter::ProcessImage(Mat image){
    Mat dst;
    resize(image, dst, cv::Size(width, height));
    return dst;
};
