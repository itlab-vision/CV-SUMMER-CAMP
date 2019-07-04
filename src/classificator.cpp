#include "classificator.h"
DnnClassificator::DnnClassificator(string Tomodel1, string Toconfig1, string Tolabels1, double scale1, float inputWidth1, float  inputHeight1, Scalar mean1, bool swapRB1) {
	
	Tomodel = Tomodel1;
	Toconfig = Toconfig1;
	Tolabels = Tolabels1;
	inputWidth = inputWidth1;
	inputHeight = inputHeight1;
	scale = scale1;
	mean = mean1;
	swapRB = swapRB1;
	

	Net net = readNet(Tomodel, Toconfig);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

}

Mat DnnClassificator::Classify(Mat image) {
	Mat inputTensor;
	Net net = readNet(Tomodel, Toconfig);
	Mat blob;


	blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
	net.setInput(blob);
	Mat prob = net.forward();
	prob.reshape(1, 1);
	return prob;
	

 }