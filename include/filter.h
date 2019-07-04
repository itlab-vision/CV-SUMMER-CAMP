#pragma once
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


class Filter
{
public:
	virtual void ProcessImage(const cv::Mat &src, cv::Mat &dst) = 0 {}
};



class GrayFilter : Filter
{
private:

public:
	void ProcessImage(const cv::Mat &src, cv::Mat &dst);

};



class ResizeFilter : Filter
{
private:
	int m_width;
	int m_height;
public:
	ResizeFilter(int width, int height);

	void ProcessImage(const cv::Mat &src, cv::Mat &dst);

};



class GaussianFilter
{
private:
	cv::Size m_kernel;
public:
	GaussianFilter(const cv::Size &kernel);

	void ProcessImage(const cv::Mat &src, cv::Mat &dst);

};