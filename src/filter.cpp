#include "filter.h"

//нет смысла конвертировать изображение RGB к BGR для чб эффекта(?).

Mat GrayFilter::ProcessImage(Mat image) {
	cvtColor(image, image, COLOR_BGR2GRAY);
	return image;
}

Mat ResizeFilter::ProcessImage(Mat image) {
	resize(image, image, Size(width, height));
	return image;

}

