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
DnnDetector::DnnDetector(Mat _image, string _model, string _config, string _labels) {
	image =_image;
	model = _model;
	config =_config;
	labels =_labels;
}
Mat DnnDetector::Detect() {
	Net net = readNet(model, config);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

	Mat inputTensor;
	blobFromImage(image, inputTensor, 0.007843, Size(300, 300), { 127.5, 127.5, 127.5 }, false, false, CV_32F);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	return prob;
}