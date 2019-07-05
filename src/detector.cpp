#include "detector.h"

DnnDetector::DnnDetector(string ptm, string ptc, string ptl, int nwidth, int nheight, Scalar nmean, bool srb,double sc) {
	path_to_model = ptm;
	path_to_config = ptc;
	path_to_labels = ptl;
	width = nwidth;
	height = nheight;
	mean = nmean;
	swap = srb;
	scale = sc;
	net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}

vector<DetectedObject> DnnDetector::Detect(Mat image) {
	Mat tens, tmp1;
	blobFromImage(image, tens, scale, Size(width, height), mean, swap, false);
	net.setInput(tens);
	tmp1 = net.forward().reshape(1, 1);
	int rows = tmp1.cols/7;
	tmp1 = tmp1.reshape(1, rows);
	vector<DetectedObject> result;
	for (int i = 0; i < rows; i++) {
		DetectedObject tmp;
		tmp.classId = tmp1.at<float>(i, 1);
		tmp.score = tmp1.at<float>(i, 2);
		tmp.Left = tmp1.at<float>(i, 3)*image.cols;
		tmp.Bottom = tmp1.at < float>(i, 4)*image.rows;
		tmp.Right = tmp1.at<float>(i, 5)*image.cols;
		tmp.Top = tmp1.at<float>(i, 6)*image.rows;
		result.push_back(tmp);
	}
	return result;
}