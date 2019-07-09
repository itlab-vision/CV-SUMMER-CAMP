#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <detector.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
"{ i image        |        | image to process         }"
"{ m model_path        |        | image to process         }"
"{ c config_path        |        | image to process         }"
"{ l label_path        |        | image to process         }"
    "{ q ? help usage |        | print help message       }";


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
  
  string imgName = parser.get<string>("image"); 
  string model = parser.get<string>("model_path");
  string config = parser.get<string>("config_path");
  string labels = parser.get<string>("label_path");

  model = "C:/Users/temp2019/Desktop/lindigas/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel";
  config = "C:/Users/temp2019/Desktop/lindigas/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.prototxt";
  labels = "C:/Users/temp2019/Desktop/lindigas/object_detection/common/mobilenet-ssd/caffe/lables.lables";
  imgName = "C:/Users/temp2019/Desktop/lindigas/data/lobachevsky.jpg";

  Mat image = cv::imread(imgName);

  Detector* detct = new DnnDetector(model, config, labels);
  vector<DetectedObject> mat = detct-> Detect(image);

  Point classIdPoint;
  double confidence;

  cv::namedWindow("Mat image", cv::WINDOW_NORMAL);
  cv::imshow("Mat image", image);
  cv::waitKey(0);
  cout << "This is empty template sample." << endl;

  return 0;
}