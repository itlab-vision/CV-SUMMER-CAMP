#include <vector>
#include "filter.h"
#include <time.h>
Mat GrayFilter::ProcessImage(Mat image) {
    cv::Mat dst;
    cvtColor(image, dst, COLOR_BGR2GRAY);
    return dst;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image) {
    cv::Size s(width, height);
    Mat dst; 
    resize(image, dst, s);
    return dst;
}

MixFilter::MixFilter(int N) {
	n = N;
}

Mat MixFilter::ProcessImage(Mat src) {
	
	int newHeight = src.cols;
	int newWidth = src.rows;

	if (src.rows % n != 0)
		newHeight = src.rows - src.rows % n;
	if (src.cols % n != 0)
		newWidth = src.cols - src.cols % n;
	Mat tmp;
	resize(src, tmp, Size(newWidth, newHeight));

	vector<Mat> parts;
	Rect roi;
	int x = 0, y = 0, roiWidth = tmp.cols / n , roiHeight = tmp.rows / n;
	for (int i = 0; i < n; i++) {
		
		
		for (int j = 0; j < n; j++) {
			roi = Rect(x, y, roiWidth, roiHeight);
			parts.push_back(tmp(roi));
			x += roiWidth;
		}
		x = 0;
		y = y + roiHeight;
		
		
	}
	//for (int i = 0; i < n*n; i++) {
	//	//imshow("part" + std::to_string(i), parts[i]);

	//}
	waitKey(0);
	vector<Mat> tmpParts(parts);
	
	srand(time(NULL));
	//std::vector<int> vector1(12, 0);
	for (int i = 0; i < n*n; i++) {
		int index = rand() % n*n;
		tmpParts[index].copyTo(parts[i]);
	}
	
	return tmp;
}