#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"
#include "classificator.cpp"

using namespace cv;
using namespace std;


int main(){
    // Load image and init parameters
    string labels_1000[1000];
    ifstream inp("/home/augustinmay/CLionProjects/practice-2/image_net.txt");
    int count=0;
    string str;
    while (std::getline(inp, str)){
        labels_1000[count] = str;
        count++;
    }
    Mat image = imread("/home/augustinmay/Downloads/space-shuttle.jpg");
    string model = "/home/augustinmay/Downloads/squeezenet1_1_caffemodel.caffemodel";
    string config = "/home/augustinmay/Downloads/squeezenet1_1_prototxt.prototxt";
    string labels = "/home/augustinmay/CV-SUMMER-CAMP/data/squeezenet1.1.labels";;
    double scale=1;
    int width = 224;
    int heigth = 224;
    Scalar mean = {104, 117, 123};
    bool swap = false;
    //Image classification
    Classificator* dnnClassificator = new DnnClassificator(model, config, labels, scale, width, heigth, mean, swap);
    Mat prob = dnnClassificator->Classify(image);
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    string text1 = "label: " + labels_1000[classId], text2 = "with confidence: " +to_string(confidence*100)+ " %";
    putText(image, text1, Point(5, 25), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 255, 255), 5);
    putText(image, text1, Point(5, 25), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 0), 2);
    putText(image, text2, Point(5, 50), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 255, 255), 5);
    putText(image, text2, Point(5, 50), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 0), 2);
    //Show result
    imshow("My image: ", image);
    waitKey(0);
    return 0;
}