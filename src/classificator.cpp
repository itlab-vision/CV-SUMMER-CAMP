#include "classificator.h"

DnnClassificator::DnnClassificator(string caffemodel1, string prototxt1, string labels1, int inputWidth1, int inputHeight1, Scalar mean1, bool swapRB1,float scale1) {

	caffemodel = caffemodel1;
	prototxt = prototxt1;
	labels = labels1;
	inputWidth = inputWidth1;
	inputHeight = inputHeight1;
	mean = mean1;
	swapRB = swapRB1;
	scale = scale1;
	
	net = readNet(caffemodel1, prototxt1);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

	return;
}

Mat DnnClassificator::Classify(Mat image) {

	Mat inputTensor;
	Net net = readNet(caffemodel, prototxt);
	Mat blob;
	//scale = "123";

	blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
	net.setInput(inputTensor);
	Mat out = net.forward();

	out = out.reshape(1, 1);

	return out;
};