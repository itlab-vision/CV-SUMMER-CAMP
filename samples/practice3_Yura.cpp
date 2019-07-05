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
  DnnDetector dd(parser.get<string>("model_path"), parser.get<string>("config_path"));
  vector<DetectedObject> v = dd.Detect(image);
  // Do something cool.
  //cout << "This is empty template sample." << endl;
  /*for (auto a : v)
	  for (int i = 0; i < 7;i++)
		  cout << a. << ;
*/

  return 0;
}