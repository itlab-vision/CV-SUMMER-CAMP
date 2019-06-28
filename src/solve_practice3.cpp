#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detector.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

struct NetParams
{
	float scale = 1.0;
	Scalar mean = Scalar(0.0, 0.0, 0.0, 0.0);
	bool swapRB = false;
	
	String framework = "";
	int inpWidth = 672;
	int inpHeight = 384;
	String model = "../../data/face-detection-adas-0001.bin";
	String config = "../../data/face-detection-adas-0001.xml";

	int backendId = 0;
	int targetId = 0;
};

int main(int argc, char** argv)
{
	string imageName = "../../data/lobachevsky.jpg";
	NetParams p;

	// Loading image
	Mat source = imread(imageName, 1);

	// Loaging Net
	DnnDetector detector = DnnDetector(p.model, p.config, p.inpWidth, p.inpHeight);


	vector<DetectedObject> shapes = detector.Detect(source);

	for (int i = 0; i < 3; i++)
	{
		Rect r = Rect(
			shapes[i].xLeft,
			shapes[i].yBottom,
			shapes[i].xRight,
			shapes[i].yTop
		);

 	rectangle(source, r, Scalar(0, 255, 255), 1, 8, 0);
	}

	imshow("detectors", source);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}