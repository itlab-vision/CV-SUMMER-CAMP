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
"{ w  width                             |        | image width for classification    }"
"{ h  height                            |        | image heigth fro classification   }"
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
	String imgName(parser.get<String>("image"));
	String model_path(parser.get<String>("model_path"));
	String config_path(parser.get<String>("config_path"));
	String labels_path(parser.get<String>("label_path"));
	Scalar scalar(parser.get<Scalar>("mean"));
	bool swap(parser.get<bool>("swap"));
	int width(parser.get<int>("w"));
	int height(parser.get<int>("h"));

	Mat image = imread(imgName);
	vector<DetectedObject> objs;
	DnnDetector detector(model_path, config_path, labels_path, 300, 300, swap, Scalar(127.5, 127.5, 127.5), 0.007843);
	objs = detector.Detect(image);
	namedWindow("window");
	for (auto i : objs)
	{
		Point p1(i.xLeftBottom, i.yLeftBottom);
		Point p2(i.xRightTop, i.yRightTop);
		Rect rect(p1, p2);
		rectangle(image, rect, (70, 10, 22),8);
		string text = "Chance: " + to_string(i.confidence*100) + "% class: " + i.classname;


		putText(image, text, p1, FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 255, 255), 0.2);
		//putText(image, text, p2, FONT_HERSHEY_DUPLEX, 1.2, Scalar(70, 10, 22), 1);

	}
	imshow("window", image);
	waitKey(0);
	cout << "This is empty template sample." << endl;

	return 0;
}