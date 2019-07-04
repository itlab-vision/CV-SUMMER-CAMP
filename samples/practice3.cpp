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
    "{ d detector     |        | detector                 }"
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

  String imgName(parser.get<String>("image")); //path
  string model(parser.get<string>("model_path"));
  string config(parser.get<string>("config_path"));
  string labels(parser.get<string>("label_path"));
  cv::Mat image = cv::imread(imgName);

  Detector* detect = new DnnDetector(image, model, config, labels);
  Mat mat = detect->Detect();



  cv::namedWindow("My image", cv::WINDOW_NORMAL);
  cv::imshow("My image", image);
  cv::waitKey(0);
  cout << "This is empty template sample." << endl;

  return 0;
}