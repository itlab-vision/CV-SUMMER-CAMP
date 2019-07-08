#include <string>
#include <iostream>
#include <fstream>
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
	"{ i image        | <none> | image to process                  }"
	"{ w  width       |        | image width for classification    }"
	"{ h  heigth      |        | image heigth fro classification   }"
	"{ model_path     |        | path to model                     }"
	"{ config_path    |        | path to model configuration       }"
	"{ label_path     |        | path to class labels              }"
	"{ mean           |        | vector of mean model values       }"
	"{ swap           |        | swap R and B channels. TRUE|FALSE }"
    "{ h ? help usage |        | print help message                }";


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
  int imgWidth(parser.get<int>("width"));
  int imgHeight(parser.get<int>("heigth"));
  String path_to_model(parser.get<String>("model_path"));
  String path_to_config(parser.get<String>("config_path"));
  String path_to_labels(parser.get<String>("label_path"));
  Scalar mean(parser.get<Scalar>("mean"));
  int swap(parser.get<int>("swap"));

  Mat image = imread(imgName);

  DnnDetector dnn(path_to_model, path_to_config, path_to_labels, imgWidth, imgHeight, swap, mean);
  vector<DetectedObject> vec = dnn.Detect(image);
  string labels[21] = {
	  "background",
	  "aeroplane",
	  "bicycle",
	  "bird",
	  "boat",
	  "bottle",
	  "bus",
	  "car",
	  "cat",
	  "chair",
	  "cow",
	  "diningtable",
	  "dog",
	  "horse",
	  "motorbike",
	  "person",
	  "pottedplant",
	  "sheep",
	  "sofa",
	  "train",
	  "tvmonitor"
  };

  for (int i = 0; i < vec.size(); i++)
  {
	  string className = labels[vec[i].uuid];
	  rectangle(image, Point(vec[i].Left, vec[i].Bottom), Point(vec[i].Right, vec[i].Top), (0, 0, 255));
	  putText(image, className, Point(vec[i].Left, vec[i].Bottom-6), FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, 0);
  }

  namedWindow("My image", WINDOW_AUTOSIZE);
  imshow("My image", image);
  waitKey(0);
  return 0;
}