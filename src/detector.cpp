#include "detector.h"
DnnDetector::DnnDetector(vector<string> classes, string model, string config, int inputWidth, int inputHeight, bool swapRB, double  scale, Scalar mean, int backendId, int targetId)
{
    this->classes = classes;
    this->model = model;
    this->config = config;
    this->inputHeight = inputHeight;
    this->inputWidth = inputWidth;
    this->swapRB = swapRB;
    this->scale = scale;
    this->mean  = mean;
    this->backendId = backendId;
    this->targetId = targetId;
    
    net = readNet(model, config);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    
};

vector<DetectedObject> DnnDetector::Detect(Mat image)
{
    vector<DetectedObject> vdet;
    Mat inputTensor;
    blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
    
    net.setInput(inputTensor);
    Mat outs = net.forward();
    outs =  outs.reshape(1, 1);
    
    int rows = outs.cols/7;
    outs = outs.reshape(1, rows);
    
    for (int i = 0; i < outs.rows; i++)
    {
        DetectedObject det;
        det.confidence = outs.at<double>(i, 2);
        //if (det.confidence > confThreshold)
        if(true)
        {

            det.uuid      = outs.at<float>(i, 1);
            det.Left      = outs.at<float>(i, 3)*image.cols;
            det.Bottom    = outs.at<float>(i, 4)*image.rows;
            det.Right     = outs.at<float>(i, 5)*image.cols;
            det.Top       = outs.at<float>(i, 6)*image.rows;
            det.classname = classes[det.uuid];
    
            vdet.push_back(det);
        }
   
    }    
    return vdet;
}
