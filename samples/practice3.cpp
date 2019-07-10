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
    "{ i image        |        | image to process           }"
    "{ m model_path   |        | path to model              }"
    "{ c config_path  |        | path to model configuration}"
    "{ l label_path   |        | path to class labels       }"
    "{ h ? help usage |        | print help message         }";


int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string imgName(parser.get<string>("image"));
	string pathToModel(parser.get<string>("model_path"));
  string pathToConfig(parser.get<string>("config_path"));
  string pathToLabels(parser.get<string>("label_path"));


  // Do something cool.
  Mat img = imread(imgName);
  DnnDetector* detector = new DnnDetector(pathToModel, pathToConfig, pathToLabels);
  vector <DetectedObject> vecOfResults = detector->Detect(img);
  for(int i=0; i < vecOfResults.size(); i++){
    Point leftBottom(vecOfResults[i].left, vecOfResults[i].bottom);
    Point rightTop(vecOfResults[i].right, vecOfResults[i].top);
    Rect rect(leftBottom, rightTop);
    rectangle(img, rect, Scalar(0, 255, 0), 2);
    cout << detector->DecodeLabels(vecOfResults[i].classId) << endl;
  }
  imshow("", img);
  waitKey();
  return 0;
}
