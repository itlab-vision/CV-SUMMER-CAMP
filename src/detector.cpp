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
DnnDetector::DnnDetector(string model_path, string config_path, string labels_path, int inputWidth, int inputHeight, bool mirror, Scalar scalar, double scale):
	model_path(model_path), config_path(config_path), labels_path(labels_path), width(inputWidth), height(inputHeight), mirror(mirror), scale(scale)
{
	this->scalar = scalar;
	net = readNet(model_path, config_path);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}
vector <DetectedObject> DnnDetector::Detect(Mat image)
{
	double thresh = 0.1;
	ParseLabels();
	Mat result;
	Mat inputTensor;
	vector <DetectedObject> retObjs;
	blobFromImage(image, inputTensor, scale, Size(width, height), scalar, mirror, false);
	net.setInput(inputTensor);
	result = net.forward();
	result = result.reshape(1, 1);
	int rows = result.cols / 7;
	result = result.reshape(1, rows);
	DetectedObject token;
	cout << rows;
	for (int i = 0; i < rows; i++)
	{
		double chance = result.at<float>(i, 2);
		cout << chance << endl;
		if (result.at<float>(i, 2) >= thresh) 
		{
			token.uuid = result.at<float>(i, 1);

			token.xLeftBottom = image.cols*result.at<float>(i, 3);// *widthFactor;
			token.yLeftBottom = image.rows*result.at<float>(i, 4);// *heightFactor;
			token.xRightTop = image.cols*result.at<float>(i, 5);// *widthFactor;
			token.yRightTop = image.rows*result.at<float>(i, 6);// *heightFactor;

			if (token.uuid > labels.size())
				token.classname = "undef";
			else
				token.classname = labels[token.uuid];
			retObjs.push_back(token);
		}
		/*uchar* ptr = result.data + i*result.step;
		token.uuid = ptr[1];
		if (token.uuid > labels.size())
			token.classname = "undef";
		else
			token.classname = labels[ptr[1]];
		token.Bottom = ptr[4];
		token.Right = ptr[5];
		token.Top = ptr[6];
		token.Left = ptr[7];
		res.push_back(token);*/
	}
	return retObjs;
}