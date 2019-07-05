#include <string>
#include <iostream>
#include <fstream>

#include "detector.h"
#include "detectedobject.h"

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
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ scale                                |        | scale                             }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";



int main(int argc, const char** argv) {
	CommandLineParser parser(argc, argv, cmdOptions);
	parser.about(cmdAbout);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
  // Load image and init parameters
  String imgName(parser.get<String>("image"));
  cv::Mat src;
  src = imread(imgName);

//Image classification
  string mp = parser.get<string>("model_path");
  string cp = parser.get<string>("config_path");
  string lp = parser.get<string>("label_path");
  int wid = parser.get<int>("width");
  int hei = parser.get<int>("heigth");
  Scalar me = parser.get<Scalar>("mean");
  int sw = parser.get<int>("swap");
  double scale = parser.get<double>("scale");
  DnnDetector det(mp, cp, lp, wid, hei, me, sw, scale);
  vector<DetectedObject> res = det.Detect(src);
  string lab[21];
  int count = 0;
  std::ifstream ifs(lp);
  while (!ifs.eof())getline(ifs, lab[count++]);

  while (!res.empty()) {
	  string objClass = "Class: " + std::to_string(res.back().classId) + " " + lab[res.back().classId];
	  string conf = "Confidence: " + std::to_string(res.back().score);
	  Point leftbottom(res.back().Left, res.back().Bottom);
	  Point righttop(res.back().Right, res.back().Top
);
	  cout << leftbottom << " " << righttop << endl;
	  Rect box(leftbottom, righttop);
	  cout << objClass << " " << conf << endl;
	  rectangle(src, box, Scalar(71, 99, 255), 1, 1, 0);

	  putText(src, objClass, Size(res.back().Left - 10, res.back().Bottom - 20), FONT_HERSHEY_COMPLEX_SMALL, 1,
		  Scalar(10, 110, 255), 1, 0);
	  putText(src, conf, Size(res.back().Right - 10, res.back().Top - 2), FONT_HERSHEY_COMPLEX_SMALL, 1,
		  Scalar(10, 99, 255), 1, 0);
	  res.pop_back();
  }
  imshow("win", src);
  waitKey(0);
  return 0;
}