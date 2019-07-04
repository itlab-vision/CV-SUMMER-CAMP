#include "classificator.h"
#include <opencv2/imgproc.hpp>

using namespace cv;

DnnClassificator::DnnClassificator(String modelPath, String configPath, String labelsPath, int inputWidth, int inputHeight, Scalar mean, bool swapRB)
{
    this->modelPath = modelPath;
    this->configPath = configPath;
    this->labelsPath = labelsPath;
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->mean = mean;
    this->swapRB = swapRB;

    net = cv::dnn::readNet(modelPath, configPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
};

Mat DnnClassificator::Classify(Mat image)
{
    Mat inputTensor;
    blobFromImage(image, inputTensor, 1, Size(inputWidth, inputHeight), mean, swapRB, false);
    net.setInput(inputTensor);
    return net.forward().reshape(1, 1);
};
