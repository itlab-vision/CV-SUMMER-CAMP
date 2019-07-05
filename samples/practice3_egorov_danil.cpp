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
	"{ model_path     |        | path to model            }"
	"{ config_path    |        | path to model configurati}"
	"{ label_path     |        | path to class labels     }"
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

  // Do something cool.
  String imgName(parser.get<String>("image"));
  String model_path(parser.get<String>("model_path"));
  String config_path(parser.get<String>("config_path"));
  String label_path(parser.get<String>("label_path"));
  //
  Mat image = imread(imgName);
  DnnDetector Detector(model_path, config_path, label_path);

  vector<DetectedObject> detectedObject = Detector.Detect(image);

  return 0;
}