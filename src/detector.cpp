#include "detector.h"
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace cv;

void Detector::SetLabels(string labelsPath)
{
    classesNames = vector<string>();
    ifstream input(labelsPath);
    string line;
    while (getline(input, line))
    {
        classesNames.push_back(line);
    }
}

vector<string> Detector::GetLabels()
{
    return classesNames;
}

DnnDetector::DnnDetector(String modelPath, String configPath, String labelsPath, float scale, int inputWidth, int inputHeight, Scalar mean, bool swapRB, float scoreThreshold)
{
    this->modelPath = modelPath;
    this->configPath = configPath;
    this->labelsPath = labelsPath;
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->scale = scale;
    this->mean = mean;
    this->swapRB = swapRB;
    this->scoreThreshold = scoreThreshold;

    SetLabels(labelsPath);

    net = cv::dnn::readNet(modelPath, configPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
};

vector<DetectedObject> DnnDetector::Detect(Mat image)
{
    Mat inputTensor;
    blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
    net.setInput(inputTensor);

    Mat outMatrix = net.forward().reshape(1, 1);
    outMatrix = outMatrix.reshape(1, outMatrix.cols / 7);

    vector<string> labels = GetLabels();
    vector<DetectedObject> outVector;

    for (int i = 0; i < outMatrix.rows; i++)
    {
        DetectedObject outObject;
        outObject.confidence = outMatrix.at<float>(i, 2);
        if (outObject.confidence >= scoreThreshold)
        {
            outObject.uuid = outMatrix.at<float>(i, 1);
            outObject.Left = outMatrix.at<float>(i, 3) * image.size().width;
            outObject.Top = outMatrix.at<float>(i, 4) * image.size().height;
            outObject.Right = outMatrix.at<float>(i, 5) * image.size().width;
            outObject.Bottom = outMatrix.at<float>(i, 6) * image.size().height;
            outObject.classname = labels[outObject.uuid - 1];
            outVector.push_back(outObject);
        }
    }
    return outVector;
};
