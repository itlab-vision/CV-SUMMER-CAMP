#include "classificator.h"

DnnClassificator::DnnClassificator(string model, string config, string labels, double scale, float inWidth, float inHeight, Scalar mean, bool swapRB) {
    this->model = model;
    this->config = config;
    this->labels = labels;
    this->scale = scale;
    this->inWidth = inWidth;
    this->inHeight = inHeight;
    this->mean = mean;
    this->swapRB = swapRB;
    net = readNet(model, config);
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);
}
Mat DnnClassificator::Classify(Mat image) {
    Size check_size = Size(inWidth, inHeight);
    Mat Tensor;
    blobFromImage(image, Tensor, scale, check_size, mean, swapRB, false, CV_32F);
    net.setInput(Tensor);
    Mat prob = net.forward();
    return prob;
}