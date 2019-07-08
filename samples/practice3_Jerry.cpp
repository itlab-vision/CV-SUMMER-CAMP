#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

//bool debug = true;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
    "{ i image        |        | image to process         }"
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

  // Do something cool. 
  Mat image = imread(imgName);

  DnnDetector detector;
  detector.Detect(image);

  /*Mat result = detector.Detect(image);

  Point classIdPoint;
  double probability;
  minMaxLoc(result.reshape(1, 1), 0, &probability, 0, &classIdPoint);
  int classId = classIdPoint.x;*/

  // для нескольких точек надо работать с MatchTemplate ?




 /* if (debug)
	  cout << "matrix: " << result.reshape(1, 1) << endl;

  cout << "probability: " << probability << ", class: " << classId << endl;*/

  system("pause");

  return 0;
}