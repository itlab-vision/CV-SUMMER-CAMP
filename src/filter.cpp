#include "filter.h"

Mat GrayFilter::ProcessImage(Mat image)
{
	cvtColor(image, image, COLOR_BGR2GRAY);
	return image;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
}

Mat ResizeFilter::ProcessImage(Mat image)
{
	resize(image, image, cv::Size(width, height));
	return image;
}

Mat GaussFilter::ProcessImage(Mat image)
{
	{
		
	
		double gauss[3][3]= { { 0.5, 0.75, 0.5 },
							{ 0.75, 1.00, 0.75 },
							{ 0.5, 0.75, 0.5 } };

		for (int y = 0; y < image.rows; y++)
		{
			for (int x = 0; x < image.cols; x++)
			{
				double R = 0.0;
				double G = 0.0;
				double B = 0.0;
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						int kx = x + j - 1;
						int ky = y + i - 1;
						if (kx < 0 || ky < 0 || kx >= image.cols || ky >= image.rows)
						{
							continue;
						}
						Vec3b intensity = image.at<Vec3b>(ky, kx);
						R += intensity.val[2] * gauss[i][j];
						G += intensity.val[1] *gauss[i][j];
						B += intensity.val[0] * gauss[i][j];
					}
				}
				R /= 6;
				G /= 6;
				B /= 6;
				image.at<Vec3b>(y, x) = Vec3d(B, G, R);
			}
		}

		return image;
	}
}

void WebCamVideo::getVideo(Filter * filter)
{
	Mat frame;
	VideoCapture cap(0);
	while (true)
	{
		cap >> frame;
		frame=filter->ProcessImage(frame);
		imshow("webcam", frame);
		if (waitKey(30) >= 0) break;
	}
}

void WebCamVideo::getVideo()
{
	Mat frame;
	VideoCapture cap(0);
	while (true)
	{
		cap >> frame;
		imshow("webcam", frame);
		if (waitKey(30) >= 0) break;
	}
}
