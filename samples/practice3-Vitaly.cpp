#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "detectedobject.h"
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
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ scale                                |        | scale                             }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";


int main(int argc, const char** argv) {
	// Process input arguments
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

  // Do something cool.
	
	String imgName(parser.get<String>("image"));
	string model = parser.get<string>("model_path");
	string config = parser.get<string>("config_path");
	string label = parser.get<string>("label_path");
	int width(parser.get<int>("width"));
	int height(parser.get<int>("heigth"));
	Scalar mean = parser.get<Scalar>("mean");
	double scale(parser.get<double>("scale"));
	bool swapRB = parser.get<int>("swap");

	Mat src = imread(imgName);
	DnnDetector  DnnDet(model, config, label, width, height, mean, scale, swapRB);

	vector <DetectedObject> detObj = DnnDet.Detect(src);
	string labels[21];
	ifstream file (label);
	int i = 0;
	while (!file.eof())
	{
		getline(file, labels[i]);
		i++;
	}
	while (!detObj.empty()) {
		string objClass = "Class: "  + labels[detObj.back().uuid];
		string conf = "Confidence: " + to_string(detObj.back().confidence);
		Point leftbottom(detObj.back().Left, detObj.back().Bottom);
		Point righttop(detObj.back().Right, detObj.back().Top
		);
		cout << leftbottom << " " << righttop << endl;
		Rect box(leftbottom, righttop);
		cout << objClass << " " << conf << endl;
		rectangle(src, box, Scalar(0, 255, 0), 2, 1, 0);

		putText(src, objClass, Size(detObj.back().Left , detObj.back().Bottom-17), FONT_HERSHEY_COMPLEX_SMALL, 1,
			Scalar(0, 255, 0), 1, 0);
		putText(src, conf, Size(detObj.back().Left , detObj.back().Bottom-3), FONT_HERSHEY_COMPLEX_SMALL, 1,
			Scalar(0, 255, 0), 1, 0);
		detObj.pop_back();
	}
	imshow("Detected", src);
	waitKey(0);
	return 0;
}