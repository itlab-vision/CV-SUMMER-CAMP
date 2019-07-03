#include "filter.h"
#include <random>

Mat GrayFilter::ProcessImage(Mat image) {
    Mat out;
    cv::cvtColor(image, out, COLOR_BGR2GRAY);
    return out;
};

ResizeFilter::ResizeFilter(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;
};

Mat ResizeFilter::ProcessImage(Mat image) {
    Mat out;
    cv::resize(image, out, cv::Size(width, height));
    return out;
}

Mat RearrangeFilter::ProcessImage(Mat image) {
	
	// Creating random sequence
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	array<short, 16> arr{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	shuffle(arr.begin(), arr.end(), default_random_engine(seed));

	// Filling the sequence with the coordinates
	vector<array<int, 4>> points;

	int widthQuarter = image.size().width / 4;
	int heightQuarter = image.size().height / 4;

	for (int y = 0; y < 16; y++)
	{
		int numX = y % 4;
		int numY = y / 4;
		array<int, 4> pointsArr = { widthQuarter * numX, heightQuarter * numY };
		points.push_back(pointsArr);
	}

	// Creating new image
	cv::Mat newImage(image.size(), image.type());
	for (int i = 0; i < 16; i++)
	{
		int startX = points.at(arr[i])[0];
		int startY = points.at(arr[i])[1];

		int newStartX = points.at(i)[0];
		int newStartY = points.at(i)[1];

		for (int y = newStartY; y < newStartY + heightQuarter; y++)
		{
			for (int x = newStartX; x < newStartX + widthQuarter; x++)
			{
				newImage.at<Vec3b>(y, x) = image.at<Vec3b>(startY + y - newStartY, startX + x - newStartX);
			}
		}
	}

	return newImage;
}
