#include "detector.h"

DnnDetector::DnnDetector(string _path_to_model, string _path_to_confing, string _path_to_labels)
{
	path_to_model = _path_to_model;
	path_to_confing = _path_to_confing;
	path_to_labels = _path_to_labels;

	net = readNet(path_to_model, path_to_confing);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}

vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	Mat inputTensor;
	vector<DetectedObject> result;
	blobFromImage(image, inputTensor, 0.007843, Size(300, 300), { 127.5, 127.5, 127.5 }, false, false, CV_32F);
	net.setInput(inputTensor);
	Mat detections = net.forward();
	detections = detections.reshape(1, 1);

	DetectedObject tmp;

	tmp.Left = detections.at<int>(1, 1);
	
	
	//for (int i = 0; i < prob.rows; i++)
	//{

	//}


	return result;
}
