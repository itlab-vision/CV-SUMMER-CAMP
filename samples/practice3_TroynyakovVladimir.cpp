#include <string>
#include <iostream>
#include "detector.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ m  model_path                           |        | path to model                     }"
"{ c  config_path                          |        | path to model configuration       }"
"{ l  label_path                           |        | path to class labels              }"
"{ h ? help usage                       |        | print help message                }";


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
  string model_path(parser.get<string>("model_path"));
  string config_path(parser.get<string>("config_path"));
  string label_path(parser.get<string>("label_path"));
  Mat image = imread(imgName);

  Detector* detector = new DnnDetector(model_path, config_path, label_path);
  vector<DetectedObject> vec = detector->Detect(image);
  for (int i = 0; i < vec.size(); i++) {
		  Point leftbottom(vec[i].left, vec[i].bottom);
		  Point righttop(vec[i].right, vec[i].top);
		  Rect rect(leftbottom, righttop);
		  rectangle(image, rect, Scalar(0, 255, 0), 2);
		  cout << "Class: " << vec[i].className << endl;
		  cout << "Confidence: " << to_string(vec[i].score) << endl;
		  cout << leftbottom << " " << righttop << endl;
		  //cout << vec[i].left << " " << vec[i].right << " " << vec[i].bottom << " " << vec[i].top;
	  
  }
  imshow("Detected", image);
  waitKey(0);
  delete detector;
  return 0;
}