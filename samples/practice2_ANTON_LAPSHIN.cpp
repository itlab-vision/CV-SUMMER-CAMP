#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             |         | image to process                  }"
"{ w  width                             |         | image width for classification    }"
"{ h  heigth                            |         | image heigth for classification   }"
"{ model_path                           |         | path to model                     }"
"{ config_path                          |         | path to model configuration       }"
"{ label_path                           |         | path to class labels              }"
"{ mean                                 | 0 0 0 0 | vector of mean model values       }"
"{ swap                                 | <FALSE> | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |         | print help message                }";

bool is_file_exist(const String fileName)
{
    ifstream file(fileName);
    return file.good();
}

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

    // Check all parameters
    if (!parser.has("image") || !parser.has("width") || !parser.has("heigth") || !parser.has("model_path") || !parser.has("config_path") || !parser.has("label_path"))
    {
        cout << "Not all required parameters provided. Required parameters are: image, width, heigth, model_path, config_path, label_path.";
        return -1;
    }

    // Load image and init parameters
    String imgName(parser.get<String>("image"));
    int imgWidth(parser.get<int>("width"));
    int imgHeigth(parser.get<int>("heigth"));
    String modelPath(parser.get<String>("model_path"));
    String configPath(parser.get<String>("config_path"));
    String labelsPath(parser.get<String>("label_path"));
    Scalar mean(parser.get<Scalar>("mean"));
    bool swap(parser.get<bool>("swap"));

    // Check all files
    if (!is_file_exist(imgName))
    {
        cout << "Unable to open image file";
        return -1;
    }
    if (!is_file_exist(modelPath))
    {
        cout << "Unable to open model file";
        return -1;
    }
    if (!is_file_exist(configPath))
    {
        cout << "Unable to open config file";
        return -1;
    }
    if (!is_file_exist(labelsPath))
    {
        cout << "Unable to open labels file";
        return -1;
    }

    Mat img = cv::imread(imgName);

    // Image classification
    Classificator* classificator = new DnnClassificator(modelPath, configPath, labelsPath, imgWidth, imgHeigth, mean, swap);
    Mat result = classificator->Classify(img);
    Point classIdPoint;
    double confidence;
    minMaxLoc(result, 0, &confidence, 0, &classIdPoint);

    // Show result
    cout << "The image is \"" << classificator->GetLabels()[classIdPoint.x] << "\" with the confidence of " << confidence * 100 << "%.";

    return 0;
}
