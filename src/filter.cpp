#include "filter.h"

Mat GrayFilter::ProcessImage(Mat src)
{
	Mat dst;
	cvtColor(src, dst, COLOR_BGR2GRAY);		//color change function
	return dst;
}

/*Public function ResizeFilter with void-type in class*/
ResizeFilter::ResizeFilter(int newWidth, int newHeignt)
{
	/*w & h - privat parameters in ResizeFilter class*/
	width = newWidth;
	height = newHeignt;
}

Mat ResizeFilter::ProcessImage(Mat src)
{
	Mat dst;
	resize(src, dst, Size(width, height));
	return dst;
}
