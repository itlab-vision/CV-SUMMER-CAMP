#include "filter.h"

void GrayFilter::ProcessImage(const cv::Mat &src, cv::Mat &dst)
{
	cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
}



ResizeFilter::ResizeFilter(std::int32_t width, std::int32_t height)
	: m_width(width), m_height(height) {}



void ResizeFilter::ProcessImage(const cv::Mat &src, cv::Mat &dst)
{
	if (m_width <= 0 || m_height <= 0)	throw;

	cv::resize(src, dst, cv::Size(m_width, m_height));
}



GaussianFilter::GaussianFilter(cv::Size kernel)
	: m_kernel(kernel) {}



void GaussianFilter::ProcessImage(const cv::Mat &src, cv::Mat &dst)
{
	if ( (!(m_kernel.width % 2) || !(m_kernel.height % 2) ) || 
		   m_kernel.width != m_kernel.height ) throw;

	GaussianBlur(src, dst, m_kernel, 0);
}



FilterBarleyBreak::FilterBarleyBreak(std::uint32_t scale) : m_scale(scale) {}



void FilterBarleyBreak::ProcessImage(const cv::Mat &src, cv::Mat &dst)
{
	try
	{
		src.copyTo(dst);

		std::uint32_t x = dst.size().width / m_scale;
		std::uint32_t y = dst.size().height / m_scale;

		for (std::uint32_t i = 0; i < 100; i++)
		{
			cv::Rect rect1(x * (rand() % m_scale), y * (rand() % m_scale), dst.size().width / m_scale, dst.size().height / m_scale);
			cv::Rect rect2(x * (rand() % m_scale), y * (rand() % m_scale), dst.size().width / m_scale, dst.size().height / m_scale);

			cv::Mat tmp;
			dst(rect2).copyTo(tmp);
			dst(rect1).copyTo(dst(rect2));
			tmp.copyTo(dst(rect1));
		}
	}
	catch (cv::Exception &e)
	{
		std::cout << e.what() << std::endl;
	}
}