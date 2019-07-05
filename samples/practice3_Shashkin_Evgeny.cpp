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
    "{ i image        | C:\\Users\\temp2019\\Desktop\\CV-SUMMER-CAMP\\data\\lobachevsky.jpg       | image to process                  }"
	"{ w  width       | 300 | image width for classification    }"
	"{ h  heigth      | 300 | image heigth fro classification   }"
	"{ model_path     | C:\\Users\\temp2019\\Desktop\\object_detection\\common\\mobilenet-ssd\\caffe\\mobilenet-ssd.caffemodel | path to model                     }"
	"{ config_path    | C:\\Users\\temp2019\\Desktop\\object_detection\\common\\mobilenet-ssd\\caffe\\mobilenet-ssd.prototxt | path to model configuration       }"
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

  Point leftBottom;
  Point rightTop;

  for (int i = 0; i < vec.size(); i++)
  {
	  leftBottom.x = image.cols*vec[i].Left;
	  leftBottom.y = image.rows*vec[i].Bottom;
	  rightTop.x = image.cols*vec[i].Right;
	  rightTop.y = image.rows*vec[i].Top;
	  rectangle(image, leftBottom, rightTop, (0, 255, 0));
  }

  return 0;
}