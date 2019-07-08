#include "detector.h"
#include "fstream"
DnnDetector::DnnDetector(string path_to_model, string path_to_config, string path_to_label, int weight, int height):w(weight),h(height)
{
	net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

	std::ifstream input(path_to_label);
	int numObj = 0;
	input.seekg(0, ios::beg);
	labels.resize(21);
	if (input.is_open())
		while (getline(input, labels[numObj]) && numObj < 20)
			numObj++;
	input.close();

}


vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	vector<DetectedObject> res;
	Mat inputTensor, imageresized;
	//resize(image, imageresized, Size(300, 300));
	blobFromImage(image, inputTensor, 0.007, Size(w, h), Scalar(127.5, 127.5, 127.5), true, false);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);
	prob = prob.reshape(1, prob.cols/7);
	DetectedObject obj;
	vector<DetectedObject> result;
	for (int i = 0; i < prob.rows; i++)
	{
		if(prob.at<float>(i,2)>=0.2)
		{
			obj.uuid = prob.at<float>(i, 1);
			obj.score = prob.at<float>(i, 2);
			obj.Left = prob.at<float>(i, 3)*image.cols;
			obj.Bottom = prob.at<float>(i, 4)*image.rows;
			obj.Right = prob.at<float>(i, 5)*image.cols;
			obj.Top = prob.at<float>(i, 6)*image.rows;
			obj.classname = labels[obj.uuid];
			result.push_back(obj);
		}
	}

	return result;
}