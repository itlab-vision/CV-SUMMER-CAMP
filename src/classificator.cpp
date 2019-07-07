#include "classificator.h"



DnnClassificator::DnnClassificator(string pathToModel, string pathToConfig, string pathToLabels,
                                   int imgWidth, int imgHeight, Scalar mean, int swapRB){
        this->pathToModel = pathToModel;
        this->pathToConfig = pathToConfig;
        this->pathToLabels = pathToLabels;
        this->imgWidth = imgWidth;
        this->imgHeight = imgHeight;
        this->mean = mean;
        this->swapRB = swapRB;
        this->net = readNet(pathToModel, pathToConfig);
        this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
        this->net.setPreferableTarget(DNN_TARGET_CPU);
}


Mat DnnClassificator::Classify(Mat image){
    Mat inputTensor;
    blobFromImage(image, inputTensor, 1.0, Size(imgWidth, imgHeight), mean, swapRB, false);
    net.setInput(inputTensor);
    Mat res = net.forward();
    res = res.reshape(1, 1);
    return res;
}

String DnnClassificator::DecodeLabel(int n) {
    ifstream fileLabels(pathToLabels);
    String line;
    while (getline(fileLabels, line)) {
        classesNames.push_back(line);
    }
    return classesNames[n];
}


