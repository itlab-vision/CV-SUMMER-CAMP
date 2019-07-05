#include "detector.h"
DnnDetector::DnnDetector(string path_to_model, string path_to_config)
{
	net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}


vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	vector<DetectedObject> res;
	Mat inputTensor, imageresized;
	//resize(image, imageresized, Size(300, 300));
	blobFromImage(image, inputTensor, 0.007, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);

	prob = prob.reshape(1, prob.cols/7);

	DetectedObject obj;
	vector<DetectedObject> result;
	for (int i = 0; i < prob.cols; i++)
	{
		Point classIdPoint;
		double confidence;
		minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);
		if(prob.at<float>(i,2)>=0.2)
		{
			obj.uuid = i;
			obj.Left = prob.at<float>(i, 3)*image.cols;
			obj.Bottom = prob.at<float>(i, 4)*image.rows;
			obj.Right = prob.at<float>(i, 5)*image.cols;
			obj.Top = prob.at<float>(i, 6)*image.rows;
			res.push_back(obj);
		}
		

	}
	//cout << prob;
	
	return res;
}