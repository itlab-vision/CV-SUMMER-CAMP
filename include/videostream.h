#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



class VideoStream {
private:
	int CamNumber;

public:
	VideoStream(int newCamNumber);
	void streamToWindow();
};