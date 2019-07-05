#include "detector.h"

DnnDetector::DnnDetector(String pathToConfing, String pathToModel, int w, int h): width(w), height(h) {
	net = readNet(pathToModel, pathToConfing);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}

vector < DetectedObject> DnnDetector::Detect(Mat image) {
	Mat blob = blobFromImage(image, 0.007, Size(width, height), Scalar(127.5, 127.5, 127.5));
	net.setInput(blob);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);
	int numberOfObjects = prob.cols / 7;
	prob = prob.reshape(1,numberOfObjects);
	vector<DetectedObject> result;
	for (int i = 0; i < prob.rows; i++)
	{
		DetectedObject obj;
		if (prob.at<float>(i, 2)>=0.2) { // confidence >=20%
			obj.uuid = i;
			obj.Left = prob.at<float>(i, 3)*image.cols;
			obj.Bottom = prob.at<float>(i, 4) * image.rows;
			obj.Right = prob.at<float>(i, 5) * image.cols;
			obj.Top = prob.at<float>(i, 6) * image.rows;
			// insert classname here
			result.push_back(obj);
		}
	}

	return result;

}
