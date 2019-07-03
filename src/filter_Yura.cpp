#include "filter.h"


Mat GrayFilter::ProcessImage(Mat image)
{
	Mat dst(image.size(), CV_8U);
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			Vec3b bgr = image.at<Vec3b>(y, x);
			uint8_t gray = (29 * bgr[0] + 150 * bgr[1] + 77 * bgr[2]) >> 8;
			dst.at<uint8_t>(y, x) = gray;
		}
	}
	return dst;
}
ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image)
{
	Mat dst;
	resize(image, dst, Size(width, height));
	return dst;
}