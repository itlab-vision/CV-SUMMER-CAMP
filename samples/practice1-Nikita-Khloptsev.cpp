#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "filter.h"
#import "filter.cpp"

using namespace cv;
using namespace std;


int main()
{
    // Load image
    GrayFilter gr;
    int w, h;
    cv::Mat src, out1, out2;
    src = imread("/home/augustinmay/Downloads/pic.jpg");
    // Filter image
    cout<<"Enter new size(width, height): ";
    cin>>w>>h;
    ResizeFilter res(w,h);
    out1 = res.ProcessImage(src);
    out2 = gr.ProcessImage(src);
    // Show image
    cv::imshow("source image", src);
    cv::imshow("filtered image", out2);
    cv::imshow("resized image", out1);
    waitKey(0);
    return 0;
}