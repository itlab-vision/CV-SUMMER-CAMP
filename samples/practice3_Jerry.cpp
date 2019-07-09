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

  String imgName(parser.get<String>("image"));
  //String imgName = "C:\\Jerry\\CV-SUMMER-CAMP\\data\\unn_neuromobile.jpg";
  // Do something cool. 
  Mat image = imread(imgName);


  bool debug = false;

  if (debug)
  {
	  imshow("123", image);
	  waitKey();
  }

  DnnDetector detector;

  vector<DetectedObject> result = detector.Detect(image);

  for (vector<DetectedObject>::iterator i = result.begin(); i != result.end(); i++)
  {
	  Point2f leftTop((*i).Left, (*i).Top);
	  Point2f rightBottom((*i).Right, (*i).Bottom);
	  Scalar scal(0, 0, 255);
	  rectangle(image, leftTop, rightBottom, scal, 5);
  }

  imshow("Detected objects", image);
  waitKey();
  /*Mat result = detector.Detect(image);

  Point classIdPoint;
  double probability;
  minMaxLoc(result.reshape(1, 1), 0, &probability, 0, &classIdPoint);
  int classId = classIdPoint.x;*/

  // для нескольких точек надо работать с MatchTemplate ?




 /* if (debug)
	  cout << "matrix: " << result.reshape(1, 1) << endl;

  cout << "probability: " << probability << ", class: " << classId << endl;*/

  return 0;
}