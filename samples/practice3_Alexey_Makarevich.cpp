#include <string>
#include <iostream>
#include<detector.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    " Hello there o/ ";

const char* cmdOptions =
"{ i image        |        | image to process			 }"
"{ model_path     |        | path to model               }"
"{ config_path    |        | path to model configuration }"
"{ label_path     |        | path to class labels        }"
"{ h ? help usage |        | print help message          }";


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
  String model_path(parser.get<String>("model_path"));
  String config_path(parser.get<String>("config_path"));
  String label_path(parser.get<String>("label_path"));
  Mat img = imread(imgName);
  //Image detection
  DnnDetector detector(model_path, config_path, label_path);
  vector<DetectedObject>  res = detector.Detect(img);
  //Show result
  Mat resImg = img;
  for (auto x : res)
  {
	  string label = x.name + ":" + to_string(x.score);
	  auto p1 = Point(x.left*img.cols, x.top*img.rows);
	  auto p2 = Point(x.right*img.cols, x.bottom*img.rows);
	  auto lablePoint = Point(x.left*img.cols+5, x.bottom*img.rows+15);
	  rectangle(resImg, p1,p2, Scalar(0,255,0),2);
	  putText(resImg, label, lablePoint, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
  }
  
  imshow("window", resImg);
  waitKey();
  return 0;
}
//-i="C:\SummerCampRepos\CV-SUMMER-CAMP\data\1.jpg" -model_path="C:\SummerCampRepos\CV-SUMMER-CAMP\data\mobilenet-ssd.caffemodel" -config_path="C:\SummerCampRepos\CV-SUMMER-CAMP\data\mobilenet-ssd.prototxt" -label_path="C:\SummerCampRepos\CV-SUMMER-CAMP\data\mobilenet.labels"