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
    ResizeFilter(int newWidth, int newHeight);
    
    Mat ProcessImage(Mat image);
    
};

class RearrangeFilter : Filter
{
private:

public:
    Mat ProcessImage(Mat image);

};

class DenoisingFilter : Filter
{
private:
    float h, hColor;
    int templateWindowSize, searchWindowSize;
public:
    DenoisingFilter(float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21);
    void setH(float h);
    void setHColor(float h);
    void setTemplateWindowSize(int h);
    void setSearchWindowSize(int h);
    float getH() const;
    float getHColor() const;
    int getTemplateWindowSize() const;
    int getSearchWindowSize() const;

    Mat ProcessImage(Mat image);

};
