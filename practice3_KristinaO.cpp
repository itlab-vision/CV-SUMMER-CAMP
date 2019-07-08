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
"{ h ? help usage |        | print help message       }"
"{ model_path     |        |                  }"
"{ config_path    |        |                  }"
"{ label_path     |        |                  }"
"{ width		  |        |                  }"
"{ height	      |        |                  }"
"{ mean		      |        |                  }"
"{ swap		      |        |                  }";

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
	
	string modelPath(parser.get<string>("model_path"));
	string configPath(parser.get<string>("config_path"));
	string labelsPath(parser.get<string>("label_path"));
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");
	Scalar mean = parser.get<Scalar>("mean");
	bool swapRB = parser.get<bool>("swap");
	Mat image = cv::imread(imgName);
	Detector* det = new DnnDetector(modelPath, configPath, labelsPath, int width, int height, Scalar mean, bool swapRB);
	vector<DetectedObject> de = det->Detect(image);
	Point classIdPoint;
	double confidence;
	for (int i = 0; i < de.size(); i++) {
		cv::rectangle(image, Point(de[i].Left, de[i].Top), Point(de[i].Right, de[i].Bottom), (255, 255, 0), 7, 5);
		cout << de[i].uuid << endl;
	}
	
	cv::namedWindow("My image", cv::WINDOW_NORMAL);
	cv::imshow("My image", image);
	cv::waitKey(0);
	// Do something cool.
	cout << "This is empty template sample." << endl;
	return 0;
}