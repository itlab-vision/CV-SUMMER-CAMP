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
"{ w  width                             |        | image width for classification    }"
"{ h  height                            |        | image heigth fro classification   }"
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
  String imgName(parser.get<String>("image"));
  String model_path(parser.get<String>("model_path"));
  String config_path(parser.get<String>("config_path"));
  String labels_path(parser.get<String>("label_path"));
  Scalar scalar(parser.get<Scalar>("mean"));
  bool swap(parser.get<bool>("swap"));
  int width(parser.get<int>("w"));
  int height(parser.get<int>("h"));

  Mat image = imread(imgName);
  vector<DetectedObject> objs;
  DnnDetector detector(model_path, config_path, labels_path, width, height, swap, scalar);
  objs = detector.Detect(image);
  cout << objs.size();
  system("pause");
  namedWindow("window");
  //for (auto i : objs)
  //{
	 // cout << i.Left << " " << i.Bottom << " "<< i.Right << " " << i.Top << " -- Rectangles (L,B,R,T)";
	 // cout << i.uuid << " -- Class id";
	 // cout << i.classname << " -- Class name" << endl;
	 // rectangle(image, Point(i.Left, i.Top), Point(i.Bottom, i.Right), (24, 162, 78), 1, 8, 0);
  //}
  imshow("window", image);
  waitKey(0);
  cout << "This is empty template sample." << endl;

  return 0;
}