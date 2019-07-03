#pragma once
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class Filter
{
    public:
    virtual Mat ProcessImage(Mat image) = 0 {}
};
class GrayFilter : Filter
{
private:

public:
    Mat ProcessImage(Mat image);	
};

class ResizeFilter : Filter
{
private:
	int width;
	int height;
    
public:
    ResizeFilter(int x, int y): width(x), height(y) {}
    Mat ProcessImage(Mat image);
	
};
