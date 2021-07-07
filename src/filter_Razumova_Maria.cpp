#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image) {
	Mat result(image.rows, image.cols, CV_8UC1);
	for (size_t i = 0; i < result.rows; i++)
	{
		for (size_t j = 0; j < result.cols; j++)
		{
			Vec3b pix = image.at<Vec3b>(i, j);
			uint8_t gray = (29 * pix[0] + 150 * pix[1] + 77 * pix[2]) >> 8;
			result.at<uint8_t>(i,j) = gray;
		}
	}
	return result;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
	width = newWidth;
	height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image) {
	Mat result;
	resize(image, result, Size(width, height));
	return result;
}