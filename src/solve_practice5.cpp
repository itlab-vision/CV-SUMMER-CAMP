#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"
#include "detector.h"
#include "tracker.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat frame;
	VideoCapture cap;

	//cap.open("../../data/faces.mp4");
	cap.open(0);
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open video\n";
		return -1;
	}

	DnnClassificator dnnClassificator = DnnClassificator(
		"../../data/squeezenet1.1.caffemodel",
		"../../data/squeezenet1.1.prototxt",
		227,
		227);
	dnnClassificator.loadClassesNames("../../data/squeezenet1.1.labels");

	DnnDetector dnnDetector = DnnDetector(
		"../../data/face-detection-adas-0001.bin",
		"../../data/face-detection-adas-0001.xml",
		672,
		384);

	while(cap.read(frame))
	{
		std::vector<DetectedObject> detectedObjects = dnnDetector.Detect(frame);
		for (int i = 0; i < 3; i++)
		{
			DetectedObject shape = detectedObjects[i];
			Mat subImage = frame.adjustROI(shape.yTop, shape.yBottom, shape.xLeft, shape.xRight);
			Mat prob = dnnClassificator.Classify(subImage);
			// Choose best class
			Point classIdPoint;
			double confidence;
			minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
			int classId = classIdPoint.x;
			shape.classname = dnnClassificator.classesNames[classId];

			rectangle(frame,
				Point(shape.xLeft, shape.yBottom),
				Point(shape.xRight, shape.yTop),
				Scalar(0, 255, 255), 1, 8, 0);
			putText(frame, shape.classname, Point(shape.xLeft, shape.yBottom), 1, 1.0, Scalar(0, 0, 255));
		}

		imshow("Pipeline", frame);
		if (waitKey(5) >= 0)
			break;
	}
	return 0;
}