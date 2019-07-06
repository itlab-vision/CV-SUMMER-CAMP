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
class GrayFilter : public Filter
{
private:

public:
<<<<<<< HEAD
	Mat ProcessImage(Mat image);

=======
   static Mat ProcessImage(Mat image);
	
>>>>>>> f1832c4255139dda91caea09b0405b018360b3a2
};

class ResizeFilter : public Filter
{
private:
	int width;
	int height;
public:
<<<<<<< HEAD
	ResizeFilter(int newWidth, int newHeight);

	Mat ProcessImage(Mat image);

};

class GaussFilter :public Filter {
public:
	Mat ProcessImage(Mat image);
};

class WebCamVideo
{
public:
	void getVideo(Filter* filter);
	void getVideo();
=======
    ResizeFilter(int newWidth, int newHeight);
	
    Mat ProcessImage(Mat image);
	
};

class GaussFilter :Filter {
public:
	static Mat ProcessImage(Mat image);
>>>>>>> f1832c4255139dda91caea09b0405b018360b3a2
};