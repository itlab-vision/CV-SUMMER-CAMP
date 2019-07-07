#include <string>
#include <iostream>
#include "detector.h"
#include "filter.h"
#include "label_parser.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
"{ i image        |        | image to process         }"
"{ h ? help usage |        | print help message       }"
"{ v video        |        | video to process         }"
"{ t type         |        | type to process          }"; //  IMAGE - 0 | VIDEO - 1 | WEBCAM - 2 |

enum types { IMAGE, VIDEO, WEBCAM };

/* PARAMETERS  */
const String modelPath = "C:/Users/rngtn/Documents/cv_summer_school/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel";
const String configPath = "C:/Users/rngtn/Documents/cv_summer_school/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.prototxt";
const String labelsPath = "C:/Users/rngtn/Documents/cv_summer_school/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/object_detection_classes.txt";
const int inputWidth = 300;
const int inputHeight = 300;
const Scalar _mean = Scalar(127.5, 127.5, 127.5);
const double scale = 1.0 / 127.5;
const bool swapRB = false;
/*END PARAMETERS*/


Mat drawDetectedObjects(Mat image, vector<DetectedObject>& objects);
void cameraDetector(VideoCapture& camera, DnnDetector& dnnDetect);


int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }



  DnnDetector detector(modelPath, configPath, labelsPath, inputWidth, inputHeight, scale, _mean, swapRB);

  int type = parser.get<int>("type");

  switch (type) {
  case types::IMAGE:
  {
	  //Initialize image
	  String imagePath = parser.get<String>("image");
	  Mat image = imread(imagePath);
	  vector<DetectedObject> detectedObjects = detector.Detect(image);
	  Mat drawenImage = drawDetectedObjects(image, detectedObjects);
	  namedWindow("Detection image", WINDOW_NORMAL);
	  imshow("Detection image", drawenImage);
	  cv::waitKey();
  }
	  break;
  case types::VIDEO:
  {
	  /* Reading */
	  String videoPath = parser.get<String>("video");
	  VideoCapture video(videoPath);

	  /* Writing */
	  double frame_width = video.get(CAP_PROP_FRAME_WIDTH);
	  double frame_height = video.get(CAP_PROP_FRAME_HEIGHT);
	  double fps = video.get(CAP_PROP_FPS); 
	  VideoWriter videoOut("C:/Users/rngtn/Documents/cv_summer_school/CV-SUMMER-CAMP/data/output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height), true);
	  Mat frame;
	  video >> frame;
	  while (video.get(CAP_PROP_POS_FRAMES) != videoOut.get(CAP_PROP_POS_FRAMES)) { // 

		  vector<DetectedObject> detectedObjects = detector.Detect(frame);
		  frame = drawDetectedObjects(frame, detectedObjects);
		  videoOut.write(frame);
		  video >> frame;
	  }
	  break;
  }
  case types::WEBCAM:
	  VideoCapture cap(0);
	  cameraDetector(cap, detector);

	  break;
  }



  return 0;
}


Mat drawDetectedObjects(Mat image, vector<DetectedObject>& objects) {
	int objectsSize = objects.size();
	for (int i = 0; i < objectsSize; i++) {
		string scoreInfo = "Score: " + to_string(objects[i].score);
		string classInfo = objects[i].classname;
		cv::rectangle(image, Rect(Point(objects[i].Left, objects[i].Top), Point(objects[i].Right, objects[i].Bottom)), Scalar(0, 128, 0));
		putText(image, classInfo, Point(objects[i].Left + 5, objects[i].Top + 24), FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 255), 1.5);
		putText(image, scoreInfo, Point(objects[i].Left + 5, objects[i].Top + 44), FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 255), 1.5);

	}

	return image;
}

void cameraDetector(VideoCapture& camera, DnnDetector& dnnDetect) {
	namedWindow("Webcam Detector", WINDOW_NORMAL);
	while (waitKey(0)) {
		Mat frame;
		camera >> frame;
		vector<DetectedObject> objects = dnnDetect.Detect(frame);
		frame = drawDetectedObjects(frame, objects);
		imshow("Webcam Detector", frame);
	}

}

