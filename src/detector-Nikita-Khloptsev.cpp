#include "detector.h"

DnnDetector::DnnDetector(string model, string config, string label, int width, int height, double scale, Scalar mean, bool swapRB){
    this->model = model;
    this->config = config;
    this->label = label;
    this->width = width;
    this->height = height;
    this->scale = scale;
    this->mean = mean;
    this->swapRB = swapRB;
    net = readNet(model, config);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
};
vector<DetectedObject> DnnDetector::Detect(Mat frame){
    Mat Tensor;
    blobFromImage(frame, Tensor, scale, Size(width, height), mean, swapRB, false, CV_32F);
    net.setInput(Tensor);
    Mat prob = net.forward().reshape(1, 1);
    int obj_num = prob.cols / 7;
    prob = prob.reshape(1, obj_num);
    vector<DetectedObject> objects;
    int cols = frame.cols;
    int rows = frame.rows;
    for (int i = 0; i < obj_num; i++){
        DetectedObject temp;
        temp.classID = prob.at<float>(i, 1);
        temp.confidence = prob.at<float>(i, 2);
        temp.Left = prob.at<float>(i, 3) * cols;
        temp.Bottom = prob.at <float>(i, 4) * rows;
        temp.Right = prob.at<float>(i, 5) * cols;
        temp.Top = prob.at<float>(i, 6) * rows;
        objects.push_back(temp);
    }
    return objects;
};