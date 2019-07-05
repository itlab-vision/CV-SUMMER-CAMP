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
"{ h  heigth                            |        | image heigth fro classification   }"
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
  String imgName(parser.get<String>("image"));
  String model_path(parser.get<String>("model_path"));
  String config_path(parser.get<String>("config_path"));
  String label_path(parser.get<String>("label_path"));
  int width(parser.get<int>("width"));
  int heigth(parser.get<int>("heigth"));
  bool swap(parser.get<bool>("swap"));
  Scalar mean(parser.get<Scalar>("mean"));

  Mat image = imread(imgName);
  DnnDetector Detector(model_path, config_path, label_path, width, heigth, mean, swap);

  vector<DetectedObject> detectedObjects = Detector.Detect(image);

  for (int i = 0; i < detectedObjects.size(); i++)
  {
	  Point leftbottom(detectedObjects.back().Left, detectedObjects.back().Right);
	  Point righttop(detectedObjects.back().Top, detectedObjects.back().Bottom);
	  Rect box(leftbottom, righttop);
	  rectangle(image, box, Scalar(65, 105, 225), 1, 1, 0);

	  cout << "Class: " << to_string(detectedObjects.back().uuid)
		  << " " << detectedObjects.back().classname << endl;
	  cout << "Score: " << to_string(detectedObjects.back().Score) << endl;
	  cout << leftbottom << " " << righttop << endl;
  }

  namedWindow("Window", WINDOW_NORMAL);
  imshow("Window", image);
  waitKey(0);

  return 0;
}