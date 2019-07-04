#include "filter.h"

void GrayFilter::ProcessImage(const cv::Mat &src, cv::Mat &dst)
{
	cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
}



ResizeFilter::ResizeFilter(int width, int height)
	: m_width(width), m_height(height) {}



void ResizeFilter::ProcessImage(const cv::Mat &src, cv::Mat &dst)
{
	if (m_width <= 0 || m_height <= 0)	throw;

	cv::resize(src, dst, cv::Size(m_width, m_height));
}



GaussianFilter::GaussianFilter(const cv::Size &kernel)
	: m_kernel(kernel) {}



void GaussianFilter::ProcessImage(const cv::Mat &src, cv::Mat &dst)
{
	if (!(m_kernel.width % 2) || !(m_kernel.height % 2)) throw;

	GaussianBlur(src, dst, m_kernel, 0);
}