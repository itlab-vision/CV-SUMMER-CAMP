#include "detector.h"

DnnDetector::DnnDetector(string pathToModel, string pathToConfig, string pathToLabels)
{
  this->pathToModel = pathToModel;
  this->pathToConfig = pathToConfig;
  this->pathToLabels = pathToLabels;

  this->net = readNet(pathToModel, pathToConfig);
  this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
  this->net.setPreferableTarget(DNN_TARGET_CPU);
}

vector<DetectedObject>DnnDetector::Detect(Mat image)
{
  Mat inputTensor;
  blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
  net.setInput(inputTensor);
  Mat output = net.forward();
  output = output.reshape(1, 1);
  output = output.reshape(1, output.cols / 7);
  vector <DetectedObject> resObj;

  for (int i = 0; i < output.rows; i++)
   {
     DetectedObject res;

     res.classId = output.at<float>(i, 1);
     res.className = this->DecodeLabels((int)output.at<float>(i, 1));
     res.score = output.at<float>(i, 2);
     res.left = output.at<float>(i, 3) * image.cols;
     res.bottom = output.at<float>(i, 4) * image.rows;
     res.right = output.at<float>(i, 5) * image.cols;
     res.top = output.at<float>(i, 6) * image.rows;

     resObj.push_back(res);
   }
  return resObj;
}

String DnnDetector::DecodeLabels(int n) {
      vector<string> classesNames;
      String line;

	    ifstream fileLabels(this->pathToLabels);
	    while (getline(fileLabels, line)) {
	        classesNames.push_back(line);
	    }
	    return classesNames[n-1];
	}
