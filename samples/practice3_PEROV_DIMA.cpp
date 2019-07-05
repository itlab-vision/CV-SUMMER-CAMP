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
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";


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
  cout << "This is empty template sample." << endl;

  String imgName(parser.get<String>("image"));
  String model_path(parser.get<String>("model_path"));
  String config_path(parser.get<String>("config_path"));
  String path_label(parser.get<String>("label_path"));
  int width(parser.get<int>("width"));
  int height(parser.get<int>("heigth"));
  Scalar mean(parser.get<Scalar>("mean"));
  int swapRB(parser.get<int>("swap"));

  DnnDetector dnn(model_path, config_path,
      path_label, width, height, mean, swapRB);

  Mat image = imread(imgName);
  Mat prob;

  //Image detecting
  vector<DetectedObject> tensor;
  tensor = dnn.Detect(image);
  
  for (int i = 0; i < tensor.size(); i++) {

      string objClass = "Class: " + std::to_string(tensor.back().uuid) + " " + tensor.back().classname;
      string conf = "Confidence: " + std::to_string(tensor.back().score);
      Point leftbottom(tensor.back().Left, tensor.back().Right);
      Point righttop(tensor.back().Top, tensor.back().Bottom);
      cout << leftbottom << " " << righttop << endl;
      Rect box(leftbottom, righttop);

      rectangle(image, box, Scalar(71, 99, 255), 1, 1, 0);


      putText(image, objClass, Size(tensor.back().Left - 10, tensor.back().Right - 20), FONT_HERSHEY_COMPLEX_SMALL, 1,
          Scalar(10, 110, 255), 1, 0);
      putText(image, conf, Size(tensor.back().Top, tensor.back().Bottom), FONT_HERSHEY_COMPLEX_SMALL, 1,
          Scalar(10, 99, 255), 1, 0);

  }


  imshow("win", image);
  waitKey(0);

  return 0;
}