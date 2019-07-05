#include <string>
#include <iostream>

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
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for detection         }"
"{ h  heigth                            |        | image heigth for detection        }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";


int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  // Do something cool.
  // load image
  String imgName(parser.get<String>("image"));
  Mat image = imread(imgName);
  resize(image, image, Size(parser.get<int>("w"), parser.get<int>("h")));

  // image detection
  DnnDetector detector(parser.get<String>("config_path"),parser.get<String>("model_path"),
	  parser.get<int>("w"), parser.get<int>("h"));
  vector<DetectedObject> detectedObjects = detector.Detect(image);
  
  // show image
  namedWindow("Show Image", WINDOW_NORMAL);
  imshow("Show Image", image);
  //rectangle(image, Point(detectedObjects[0].Left, detectedObjects[0].Top), Point(detectedObjects[0].Left + image.cols - detectedObjects[0].Right, 150));
  waitKey();

  cout << "This is empty template sample." << endl;

  return 0;
}