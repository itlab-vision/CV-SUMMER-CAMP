#include <string>
#include <iostream>
#include "detector.h"
#include "filter.h"
#include "label_parser.h"

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
    "{ i image        |        | image to process         }"
    "{ h ? help usage |        | print help message       }";


Mat drawDetectedObjects(Mat image, vector<DetectedObject>& objects) {
	int objectsSize = objects.size();
	for (int i = 0; i < objectsSize; i++) {
		string information = "Score: " + to_string(objects[i].score) + ". Class: " + to_string(objects[i].uuid);
		cv::rectangle(image, Rect(Point(objects[i].Left, objects[i].Top), Point(objects[i].Right, objects[i].Bottom)), Scalar(0, 128, 0));
		putText(image, information, Point(objects[i].Left + 5, objects[i].Top - 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
	}
	
	return image;
}

void cameraDetector(VideoCapture& camera, DnnDetector& dnnDetect) {
	namedWindow("Webcam Detector", WINDOW_NORMAL);
	while(waitKey(0)){
		Mat frame;
		camera >> frame;
		vector<DetectedObject> objects = dnnDetect.Detect(frame);
		frame = drawDetectedObjects(frame, objects);
		imshow("Webcam Detector", frame);
	}

}


int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  label

  return 0;

 String imagePath = parser.get<String>("image");

  // parameters
  const String modelPath = "C:/Users/temp2019/summer-camp/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel";
  const String configPath = "C:/Users/temp2019/summer-camp/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.prototxt";
  const int inputWidth = 300;
  const int inputHeight = 300;
  const Scalar mean = Scalar(127.5, 127.5, 127.5);
  const double scale = 1.0 / 127.5;
  const bool swapRB = false;


  //Initialize image
  Mat image = imread(imagePath);



  DnnDetector detector(modelPath, configPath, "", inputWidth, inputHeight, scale, mean, swapRB);
  vector<DetectedObject> detectedObjects = detector.Detect(image);


  Mat drawenImage = drawDetectedObjects(image, detectedObjects);

  namedWindow("Detection image", WINDOW_NORMAL);
  imshow("Detection image", drawenImage);
  cv::waitKey();
 
  VideoCapture cap(0);
  cameraDetector(cap, detector);

  return 0;
}