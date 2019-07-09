#include "detector.h"


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
	double scale = 1.0/ 127.50223128904757;

	Mat inputTensor;
	blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB /* ,0 ,0 */);
	net.setInput(inputTensor);
	Mat blob = net.forward();

	
	Mat temp = blob.reshape(1, 1); //чтобы понять ко скольки строкам приводить оригинальный блоб
	int columns = temp.cols / 7;

	bool debug = true;
	blob = blob.reshape(1, columns);
	
	vector<DetectedObject> result;	
	if (debug)
		cout << blob << endl;

	
	for (int i = 0; i < blob.rows; ++i)
	{
		DetectedObject temp;
		temp.Left = blob.at<float>(i, 3) * image.cols;
		temp.Bottom = blob.at<float>(i, 4) * image.rows;
		temp.Right = blob.at<float>(i, 5) * image.cols;
		temp.Top = blob.at<float>(i, 6) * image.rows;
		result.push_back(temp);
	}
		

	if (debug)
		cout << "result: Left:"/*1*/ << result[0].Left << " Bottom:" << result[0].Bottom << " Right:" << result[0].Right << "Top:" << result[0].Top << endl;

	


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

	return result;
}