#include <string>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    "Sample image detection application.";

const char* cmdOptions =
    "{ i image         |         | image to process                                   }"
    "{ w  width        |         | image width for detection                          }"
    "{ h  heigth       |         | image heigth for detection                         }"
    "{ s  scale        |   1     | image scale for detection                          }"
    "{ model_path      |         | path to model                                      }"
    "{ config_path     |         | path to model configuration                        }"
    "{ label_path      |         | path to class labels                               }"
    "{ mean            | 0 0 0 0 | vector of mean model values                        }"
    "{ swap            | <FALSE> | swap R and B channels. TRUE|FALSE                  }"
    "{ score_threshold |  0.75   | set a threshold for output confidence values (0-1) }"
    "{ q ? help usage  |         | print help message                                 }";

bool isFileExist(const String fileName)
{
    ifstream file(fileName);
    return file.good();
}

int main(int argc, const char** argv) {

    // Parse command line arguments.
    CommandLineParser parser(argc, argv, cmdOptions);
    parser.about(cmdAbout);
    
    // If help option is given, print help message and exit.
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
    
    String imgName(parser.get<String>("image"));
    int imgWidth(parser.get<int>("width"));
    int imgHeigth(parser.get<int>("heigth"));
    float scale(parser.get<float>("scale"));
    String modelPath(parser.get<String>("model_path"));
    String configPath(parser.get<String>("config_path"));
    String labelsPath(parser.get<String>("label_path"));
    Scalar mean(parser.get<Scalar>("mean"));
    bool swap(parser.get<bool>("swap"));
    float scoreThreshold(parser.get<float>("score_threshold"));
    
    // Check all files
    if (!isFileExist(imgName))
    {
      cout << "Unable to open image file";
      return -1;
    }
    if (!isFileExist(modelPath))
    {
        cout << "Unable to open model file";
        return -1;
    }
    if (!isFileExist(configPath))
    {
        cout << "Unable to open config file";
        return -1;
    }
    if (!isFileExist(labelsPath))
    {
      cout << "Unable to open labels file";
      return -1;
    }
    
    Mat img = cv::imread(imgName);
    
    // Image detection
    Detector* detector = new DnnDetector(modelPath, configPath, labelsPath, scale, imgWidth, imgHeigth, mean, swap, scoreThreshold);
    vector<DetectedObject> detectedObjects = detector->Detect(img);
    
    // Processing image to show result
    for (int i = detectedObjects.size() - 1; i >= 0 ; i--)
    {
        Point point1(detectedObjects[i].Left, detectedObjects[i].Top);
        Point point2(detectedObjects[i].Right, detectedObjects[i].Bottom);
        Rect box(point1, point2);

        srand(time(0) * (i + 1));
        int r = (rand() % 150) + 50;
        int g = (rand() % 150) + 50;
        int b = (rand() % 150) + 50;
        Scalar color = Scalar(r, g, b);

        rectangle(img, box, color, 4);
        
        ostringstream confidencePercents;
        confidencePercents.precision(2);
        confidencePercents << fixed << detectedObjects[i].confidence * 100;

        string text = detectedObjects[i].classname + " (" + confidencePercents.str() + "%)";

        // Check if text overflows
        int offsetX = 0;
        int offsetY = 10;
        if (text.length() * 21 > img.size().width - point1.x)
        {
            offsetX = text.length() * 21 - (img.size().width - point1.x);
        }
        if (20 + offsetY > point1.y)
        {
            offsetY = point1.y - 20 - offsetY;
        }

        putText(img, text, Point(point1.x - offsetX, point1.y - offsetY), FONT_HERSHEY_DUPLEX, 1.2, Scalar(255, 255, 255), 3);
        putText(img, text, Point(point1.x - offsetX, point1.y - offsetY), FONT_HERSHEY_DUPLEX, 1.2, color, 1);
    }

    // Show result
    cv::imshow("Result", img);
    cv::waitKey();

    return 0;
}
