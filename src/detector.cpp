#include "detector.h"

bool debug = true;

DnnDetector::DnnDetector()
{
	this->path_to_model = "C:\\Most important C\\Programms C\\Neuro Networks\\object_detection\\common\\mobilenet-ssd\\caffe\\mobilenet-ssd.caffemodel";
	this->path_to_config = "C:\\Most important C\\Programms C\\Neuro Networks\\object_detection\\common\\mobilenet-ssd\\caffe\\mobilenet-ssd.prototxt";
	this->path_to_labels = "C:\\Most important C\\Programms C\\Neuro Networks\\squeezenet1.1\\classification\\squeezenet\\1.1\\caffe\\squeezenet1.1.labels";
	this->inputWidth = 300;
	this->inputHeight = 300;
	this->mean = (127.5, 127.5, 127.5);
	this->swapRB = false;

	this->net = readNet(path_to_model, path_to_config, path_to_labels);
	//net.setPreferableBackend();
	//net.setPreferableTarget();
}

vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	double scale = 127.50223128904757;

	Mat inputTensor;
	blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB /* ,0 ,0 */);
	net.setInput(inputTensor);
	Mat blob = net.forward();

	blob = blob.reshape(1, 100);
	
	vector<DetectedObject> result;
	for (int i = 0; i < blob.rows; ++i)
	{
		DetectedObject temp;
		temp.Left = blob.cols;
			top
			right
			bottom
		
	}
		

	

	if (debug)
		cout << blob << endl;


	//for (int i = 0; i < 100; i++)
	//{
	//	for (int j = 0; j < 7; j++)
	//	{
	//		result[i].Bottom
	//	}
	//}

	


	/*
	Point classIdPoint;
	double probability;
	minMaxLoc(result.reshape(1, 1), 0, &probability, 0, &classIdPoint);
	int classId = classIdPoint.x;
	*/

	return vector<DetectedObject>();
}