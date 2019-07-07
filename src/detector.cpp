#include "detector.h"




DnnDetector::DnnDetector(String _modelPath, String _configPath, String _labelsPath, int _inputWidth, int _inputHeight, double _scale, Scalar _mean, bool _swapRB){
	modelPath = _modelPath;
	configPath = _configPath;
	labelsPath = _labelsPath;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	scale = _scale;
	mean = _mean;
	swapRB = _swapRB;

	net = readNet(modelPath, configPath);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}


vector<DetectedObject> DnnDetector::Detect(Mat image) {

	int width_source = image.cols; 
	int height_source = image.rows; // Размеры исходного изображения

	Mat inputTensor;
	vector<DetectedObject> objects;
	blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);



	net.setInput(inputTensor);
	Mat objectsProbability = net.forward().reshape(1,1);
	int objectsNum = objectsProbability.cols / 7;

	/* Getting class labels*/
	std::map<int, string> classes = initializeClasses("C:/Users/rngtn/Documents/cv_summer_school/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/object_detection_classes.txt");


	for (int i = 0; i < objectsNum; i++) {
		DetectedObject obj;
		obj.uuid = objectsProbability.at<float>(0, 1 + i * 7);
		obj.score = objectsProbability.at<float>(0, 2 + i * 7);
		obj.Left = objectsProbability.at<float>(0, 3 + i * 7) * width_source;
		obj.Top = objectsProbability.at<float>(0, 4 + i * 7) * height_source;
		obj.Right = objectsProbability.at<float>(0, 5 + i * 7) * width_source;
		obj.Bottom = objectsProbability.at<float>(0, 6 + i * 7) * height_source;



		obj.classname = classes[obj.uuid];

		objects.push_back(obj);
	}
	return objects;
}