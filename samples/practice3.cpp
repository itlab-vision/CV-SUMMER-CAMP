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

  String imgName(parser.get<String>("image")); //path
  string model(parser.get<string>("model_path"));
  string config(parser.get<string>("config_path"));
  string labels(parser.get<string>("label_path"));
  model = "C:\\Users\\world\\CV-SUMMER-CAMP\\data\\new\\object_detection\\common\\mobilenet-ssd\\caffe\\mobilenet-ssd.caffemodel";
  config = "C:\\Users\\world\\CV-SUMMER-CAMP\\data\\new\\object_detection\\common\\mobilenet-ssd\\caffe\\mobilenet-ssd.prototxt";
  labels = "C:\\Users\\world\\CV-SUMMER-CAMP\\data\\mobilenet-ssd.labels";
  imgName = "C:\\Users\\world\\CV-SUMMER-CAMP\\data\\unn_neuromobile.jpg.jpg";
  Mat image = cv::imread(imgName);
  Detector* detect = new DnnDetector(model, config, labels);
  vector<DetectedObject> mat = detect->Detect(image);
  Point classIdPoint;
  double confidence;
  for (int i = 0; i < mat.size(); i++) {
	  cv::rectangle(image, Point(mat[i].Left, mat[i].Top), Point(mat[i].Right, mat[i].Bottom), (255, 255, 255), 3);
  }
  cv::namedWindow("My image", cv::WINDOW_NORMAL);
  cv::imshow("My image", image);
  cv::waitKey(0);
  // Do something cool.
  cout << "This is empty template sample." << endl;

  return 0;
}