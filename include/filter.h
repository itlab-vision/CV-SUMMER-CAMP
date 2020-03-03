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



class GrayFilter : public Filter
{
public:
	void ProcessImage(const cv::Mat &src, cv::Mat &dst);

};



class ResizeFilter : public Filter
{
public:
	ResizeFilter(std::int32_t width, std::int32_t height);

	void ProcessImage(const cv::Mat &src, cv::Mat &dst);

private:
	std::int32_t m_width;
	std::int32_t m_height;

};



class GaussianFilter : public Filter
{
public:
	GaussianFilter(cv::Size kernel);

	void ProcessImage(const cv::Mat &src, cv::Mat &dst);

private:
	cv::Size m_kernel;

};



class FilterBarleyBreak : public Filter
{
public:
	FilterBarleyBreak(std::uint32_t scale);

	void ProcessImage(const cv::Mat &src, cv::Mat &dst);

private:
	std::uint32_t m_scale;

};