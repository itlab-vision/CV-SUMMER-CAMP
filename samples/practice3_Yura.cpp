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
    "{ i image        |        | image to process         }"
	"{ w  width                             |        | image width for classification    }"
	"{ h  heigth                            |        | image heigth fro classification   }"
	"{ model_path                           |        | path to model                     }"
	"{ config_path                          |        | path to model configuration       }"
	"{ label_path                           |        | path to model configuration       }"
    "{ h ? help usage |        | print help message       }";


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
  Mat image = imread(imgName);
  DnnDetector dd(parser.get<string>("model_path"), parser.get<string>("config_path"),parser.get<string>("label_path"), parser.get<int>("w"), parser.get<int>("h"));
  vector<DetectedObject> v = dd.Detect(image);
  for (auto a : v) {
	  string classes = "Class: " + std::to_string(a.uuid) + " " + a.classname;
	  string confidence = "Confidence: " + std::to_string(a.score);
	  Rect rect(Point(a.Left, a.Bottom), Point(a.Right, a.Top));
	  cout << classes << " " << confidence << endl;
	  rectangle(image, rect, Scalar(0, 255, 0), 1, 1, 0);

	  putText(image, classes, Size(a.Left, a.Bottom - 200), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 255, 255), 1, 0);
	  putText(image, confidence, Size(a.Left, a.Bottom - 20), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 99, 255), 1, 0);
  }
  imshow("detecting", image);
  waitKey();

  return 0;
}