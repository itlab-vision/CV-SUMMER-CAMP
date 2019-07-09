#include "detector.h"
bool DnnDetector::ParseLabels()
{
	string input;
	ifstream in(labels_path);
	bool isOpen = 0;
	if (in.is_open())
	{
		while (getline(in, input))
		{
			labels.push_back(input);
		}
		isOpen = 1;
	}
	return isOpen;

}
DnnDetector::DnnDetector(string model_path, string config_path, string labels_path, int inputWidth, int inputHeight, bool mirror, Scalar scalar, double scale) :
	model_path(model_path), config_path(config_path), labels_path(labels_path), width(inputWidth), height(inputHeight), mirror(mirror), scale(scale)
{
	this->scalar = scalar;
	ParseLabels();
	net = readNet(model_path, config_path);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}
vector <DetectedObject> DnnDetector::Detect(Mat image)
{
	double thresh = 0.5;
	Mat inputTensor;
	Mat output;
	vector <DetectedObject> retObjs;

	blobFromImage(image, inputTensor, scale, Size(width, height), scalar, mirror, false);
	net.setInput(inputTensor);

	output = net.forward();
	output = output.reshape(1, 1);
	output = output.reshape(1, output.cols/7);

	DetectedObject token;
	for (int i = 0; i < output.rows; i++)
	{
		double chance = output.at<float>(i, 2);
		cout << chance*100 << endl;
		if (output.at<float>(i, 2) >= thresh)
		{
			token.uuid = output.at<float>(i, 1);
			token.confidence = output.at<float>(i, 2);
			token.xLeftBottom = image.cols*output.at<float>(i, 3);
			token.yLeftBottom = image.rows*output.at<float>(i, 4);
			token.xRightTop = image.cols*output.at<float>(i, 5);
			token.yRightTop = image.rows*output.at<float>(i, 6);

			if (token.uuid > labels.size())
				token.classname = "undef";
			else
				token.classname = labels[token.uuid];
			retObjs.push_back(token);
		}
	}
	return retObjs;
}