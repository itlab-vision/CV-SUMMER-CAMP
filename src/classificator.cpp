#include "classificator.h"

DnnClassificator::DnnClassificator(string ptm, string ptc, string ptl, int nwidth, int nheight, Scalar nmean, bool srb) {
	path_to_model = ptm;
	path_to_config = ptc;
	path_to_labels = ptl;
	width = nwidth;
	height = nheight;
	mean = nmean;
	swap = srb;
	net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}

Mat DnnClassificator::Classify(Mat image) {
	Mat tens, res;
	blobFromImage(image, tens, 1, Size(width, height), mean, swap, false);
	net.setInput(tens);
	res = net.forward();
	res=res.reshape(1, 1);
	return res;
}