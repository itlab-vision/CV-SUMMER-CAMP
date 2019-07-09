#include <string>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "detector.h"
#include "detector.cpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main() {
    // Parse command line arguments.

    // Load image and init parameters
    Mat image = imread("/home/augustinmay/Downloads/plane.jpg");
    String model="/home/augustinmay/Downloads/mobilenet-ssd/mobilenet-ssd.caffemodel";
    String config="/home/augustinmay/Downloads/mobilenet-ssd/mobilenet-ssd.prototxt";
    String label="/home/augustinmay/Downloads/mobilenet-ssd/object_detection_classes.txt";
    int width = 300;
    int heigth = 300;
    double scale = 1.0 / 127.5;
    Scalar mean = Scalar(127.5, 127.5, 127.5);
    bool swapRB = false;
    string labels[20];
    ifstream inp(label);
    int count=0;
    string str;
    while (std::getline(inp, str)){
        labels[count] = str;
        count++;
    }
    //Image detection
    Detector *det = new DnnDetector(model, config, label, width, heigth, scale, mean, swapRB);
    vector<DetectedObject> objects = det->Detect(image);
    //Show result
    for (int i = 0; i < objects.size(); i++){
        if((objects[i].classID-1)==-1){
            break;
        }
        Point point1(objects[i].Left, objects[i].Bottom);
        Point point2(objects[i].Right, objects[i].Top);
        rectangle(image, point1, point2, Scalar(0, 0, 255), 3);
        string text1 = "label: " + labels[objects[i].classID-1]+" with confidence: " +to_string(objects[i].confidence*100)+ " %";
        putText(image, text1, point1+Point(0, -10), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 255, 255), 5);
        putText(image, text1, point1+Point(0, -10), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 0), 2);
    }
    imshow("Objects", image);
    waitKey(0);
    return 0;
}