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
    "{ i image                              |        | image to process                  }"
	"{ w  width                             |        | image width for classification    }"
	"{ h  heigth                            |        | image heigth fro classification   }"
	"{ model_path                           |        | path to model                     }"
	"{ config_path                          |        | path to model configuration       }"
	"{ label_path                           |        | path to class labels              }"
	"{ mean                                 |        | vector of mean model values       }"
	"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
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

  // Load image and init parameters
  String imgName(parser.get<String>("image"));
  Mat image = imread(imgName);

  String modelPath(parser.get<String>("model_path"));
  String configPath(parser.get<String>("config_path"));
  String labelPath(parser.get<String>("label_path"));
  int width = parser.get<int>("width");
  int heigth = parser.get<int>("heigth");
  double scale = parser.get<double>("scale");
  Scalar mean = parser.get<Scalar>("mean");
  bool swap = parser.get<bool>("swap");

  String label[21] = { "background", "aeroplane", "bicycle", "bird",
	"boat","bottle", "bus",  "car", "cat", "chair", "cow",
	"diningtable",  "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep",  "sofa", "train", "tvmonitor" };

  //Image detection
  Detector* a = new DnnDetector(modelPath, configPath, labelPath, width, heigth, scale, mean, swap);
  vector<DetectedObject> objects = a->Detect(image);

  //Show result
  for (int i = 0; i < objects.size(); i++)
  {
	  Point point1(objects[i].Left, objects[i].Bottom);
	  Point point2(objects[i].Right, objects[i].Top);
	  rectangle (image, point1, point2, Scalar(0,0,255), 3);
	  string text = "Class: " + label[objects[i].classid] + " Confidence: " + std::to_string(objects[i].score);
	  putText(image, text, Point(objects[i].Left, objects[i].Bottom - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0));
  }
  namedWindow("Objects", WINDOW_NORMAL);
  imshow("Objects", image);
  waitKey(0);

  return 0;
}