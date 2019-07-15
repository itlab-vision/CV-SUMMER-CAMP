#include <string>
#include <iostream>

#include <fstream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
    "{ i image        |        | image to process                   }"
    "{ w  width       |        | image width for classification     }"
    "{ h  heigth      |        | image heigth fro classification    }"
    "{ model_path     |        | path to model                      }"
    "{ config_path    |        | path to model configuration        }"
    "{ classes        |        | path to classes file               }"
    "{ mean           |        | vector of mean model values        }"
    "{ swap           |        | swap R and B channels. TRUE|FALSE  }"
    "{ scale          |        | scale the image for blob           }"
    "{ h ? help usage |        | print help message                 }";


void drawDetections(vector<DetectedObject>, Mat&);

int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);

  // If help option is given, print help message and exit.
    if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
    //Take classes from file
    vector<string> classes;
    string file = parser.get<String>("classes");
    ifstream ifs(file);
    string line;
    while (getline(ifs, line))
    {
        classes.push_back(line);
    }
    //Load image
    String imgName(parser.get<String>("image"));
    Mat image = imread(imgName);
    
    //Parsing parameters
    string model = parser.get<string>("model_path");
    string config = parser.get<string>("config_path");
    int inputHeight = parser.get<int>("w");
    int inputWidth = parser.get<int>("h");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("swap");
    double scale = parser.get<double>("scale");

    //Crete new detector from class and draw its result
    Detector* detector = new DnnDetector(classes, model, config, inputWidth, inputHeight, swapRB, scale, mean);
    vector<DetectedObject> detected_objects = detector->Detect(image);
    
    drawDetections(detected_objects, image);
    
    namedWindow("detected_image", WINDOW_NORMAL);
    imshow("detected_image", image);
    waitKey();
    
  return 0;
}

void drawDetections(vector<DetectedObject> detected_objects, Mat& image)
{
    auto beg = detected_objects.begin();
    for (auto it = beg; it != detected_objects.end(); ++it){
        //New box with detection
        rectangle(image,
                  Point(it->Left, it->Top),
                  Point(it->Right, it->Bottom),
                  Scalar(0, 255, 0));
        //Draw classname and confidence on the white background near detected box
        string label = format("%.5f", it->confidence);
        label = it->classname + ": " + label;
        
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top;
        top = max(it->Top, labelSize.height);
        rectangle(image,
                  Point(it->Left, top - labelSize.height),
                  Point(it->Left + labelSize.width, it->Top + baseLine),
                  Scalar::all(255), FILLED);
        putText(image, label, Point(it->Left, it->Top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
    }
}
