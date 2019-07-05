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
"{ i  image                             | <none> | image to process                  }"
"{ v  video                             | <none> | video to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ scale                                |        | scale for DNN                     }"
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
	// Load image and init parameters
	String vidName(parser.get<String>("video"));
	String modelPath = parser.get<String>("model_path");
	String configPath = parser.get<String>("config_path");
	String labelsPath = parser.get<string>("label_path");
	int width = parser.get<int>("width");
	int height = parser.get<int>("heigth");
	Scalar mean = parser.get<Scalar>("mean");
	bool swapRB = parser.get<bool>("swap");
	double scale = parser.get<double>("scale");
	DnnDetector detector(modelPath, configPath, labelsPath, width, height, mean, swapRB, scale);
	
	// Do something cool.
	if (parser.has("image")) {
		String imgName(parser.get<String>("image"));
		Mat image = imread(imgName);
		vector<DetectedObject> res;
		res = detector.Detect(image);

		while (!res.empty()) {
			string objClass = "Class: " + std::to_string(res.back().uuid) + " " + res.back().classname;
			string conf = "Confidence: " + std::to_string(res.back().score);
			Point leftbottom(res.back().xLeftBottom, res.back().yLeftBottom);
			Point righttop(res.back().xRightTop, res.back().yRightTop);
			cout << leftbottom << " " << righttop << endl;
			Rect box(leftbottom, righttop);
			cout << objClass << " " << conf << endl;
			rectangle(image, box, Scalar(71, 99, 255), 1, 1, 0);

			putText(image, objClass, Size(res.back().xLeftBottom - 10, res.back().yLeftBottom - 20), FONT_HERSHEY_COMPLEX_SMALL, 1,
				Scalar(10, 110, 255), 1, 0);
			putText(image, conf, Size(res.back().xLeftBottom - 10, res.back().yLeftBottom - 2), FONT_HERSHEY_COMPLEX_SMALL, 1,
				Scalar(10, 99, 255), 1, 0);
			res.pop_back();
		}
		imshow("win", image);
		waitKey(0);
	}
	else {
		VideoCapture cap(vidName);
		Mat frame;
		vector<DetectedObject> res;

		if (!cap.isOpened())  // check if we succeeded
			return -1;

		while (cap.isOpened()) {
			cap >> frame;
			res = detector.Detect(frame);
			while (!res.empty()) {
				string objClass = "Class: " + std::to_string(res.back().uuid) + " " + res.back().classname;
				string conf = "Confidence: " + std::to_string(res.back().score);
				Point leftbottom(res.back().xLeftBottom, res.back().yLeftBottom);
				Point righttop(res.back().xRightTop, res.back().yRightTop);
				cout << leftbottom << " " << righttop << endl;
				Rect box(leftbottom, righttop);
				cout << objClass << " " << conf << endl;
				rectangle(frame, box, Scalar(71, 99, 255), 1, 1, 0);

				putText(frame, objClass, Size(res.back().xLeftBottom - 10, res.back().yLeftBottom - 20), FONT_HERSHEY_COMPLEX_SMALL, 1,
					Scalar(10, 110, 255), 1, 0);
				putText(frame, conf, Size(res.back().xLeftBottom - 10, res.back().yLeftBottom - 2), FONT_HERSHEY_COMPLEX_SMALL, 1,
					Scalar(10, 99, 255), 1, 0);
				res.pop_back();
				imshow("win", frame);
			}
			char q = waitKey(1);
			if (q == 27) break;
		}
		cap.release();
	}
	return 0;
}