#include "videostream.h"
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

VideoStream::VideoStream(int newCamNumber) {
	CamNumber = newCamNumber;
}

void VideoStream::streamToWindow() {
	VideoCapture cap(CamNumber);
	Mat frame;
	namedWindow("Video stream", WINDOW_NORMAL);
	while (1) {
		cap.read(frame);
		imshow("Video stream", frame);
		if (waitKey(5) >= 0) {
			break;
		}
	}

}