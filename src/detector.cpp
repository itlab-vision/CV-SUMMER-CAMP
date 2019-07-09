#include "detector.h"
//ssd_caffe:
//	 model: "MobileNetSSD_deploy.caffemodel"
//		 config : "MobileNetSSD_deploy.prototxt"
//		 mean : [127.5, 127.5, 127.5]
//		 scale : 0.007843
//		 width : 300
//		 height : 300
//		 rgb : false
//		 classes : "object_detection_classes_pascal_voc.txt"
//		 sample : "object_detection"

//
//int Left;
//int Right;
//int Top;
//int Bottom;
//int uuid;
//std::string classname;

//[image_number, classid, score, left, bottom, right, top],

DnnDetector::DnnDetector(string _model, string _config, string _labels) {
	model = _model;
	config =_config;
	labels =_labels;
}
vector<DetectedObject> DnnDetector::Detect(Mat _mat) {
	Mat image = _mat;
	Net net = readNet(model, config);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

	Mat inputTensor;
	blobFromImage(image, inputTensor, 0.007843, Size(300, 300), { 127.5, 127.5, 127.5 }, false, false);
	net.setInput(inputTensor);
	Mat prob = net.forward(); 
	prob = prob.reshape(1,1);
	//for (int i = 0; i < prob.cols; i++) std::cout << to_string(prob.at<int>(0,i))<< " / ";
	prob = prob.reshape(1, prob.cols / 7);
	vector<DetectedObject> res;
	for (int i = 0; i < prob.rows; i++) {
		DetectedObject obj;
		obj.Left = prob.at<float>(i, 3)*image.cols;
		obj.Right = prob.at<float>(i, 5)*image.cols;
		obj.Top = prob.at<float>(i, 6)*image.rows;
		obj.Bottom = prob.at<float>(i, 4)*image.rows;
		obj.classid = (int)round(prob.at<float>(i, 1));
		obj.confidence = prob.at<float>(i, 2);
		res.push_back(obj);
	}
	return res;
}