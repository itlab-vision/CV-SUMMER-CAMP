#include "classificator.h"
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";

int main(int argc, char** argv)
{
    // Process input arguments
    CommandLineParser parser(argc, argv, cmdOptions);
    parser.about(cmdAbout);
    
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    
    // Load image and init parameters
    String imgName(parser.get<String>("image"));
    int imgWidth(parser.get<int>("width"));
    int imgHeight(parser.get<int>("height"));
    String pathToModel(parser.get<String>("model_path"));
    String pathToConfig(parser.get<String>("config_path"));
    String pathToLabel(parser.get<String>("label_path"));
    Scalar mean(parser.get<Scalar>("mean"));
    int swap(parser.get<int>("swap"));
    
    //Image classification
    Mat image = imread(imgName);
    imshow("", image);
    waitKey();
    DnnClassificator* classificator = new DnnClassificator(pathToModel, pathToConfig, pathToLabel, imgWidth, imgHeight, mean, swap);
    
    Mat res = classificator->Classify(image);
    
    Point classIdPoint;
    double confidence;
    minMaxLoc(res, 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    
    //Show result
    cout << "The image is " << classificator->DecodeLabel(classId) << " with confidence " << confidence * 100 << "%";
    delete classificator;
    
    
    return 0;
}
