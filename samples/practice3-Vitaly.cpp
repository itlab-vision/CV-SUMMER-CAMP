#include <string>
#include <iostream>

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
	for (int i = 0; i < detObj.size(); i++)
	{
		Point leftbottom(detObj.back().Left, detObj.back().Right);
		Point righttop(detObj.back().Top, detObj.back().Bottom);
		Rect box(leftbottom, righttop);
		rectangle(src, box, Scalar(65, 105, 225), 1, 1, 0);

		cout << "Class: " << to_string(detObj.back().uuid)
			<< " " << detObj.back().classname << endl;
		cout << "Confidence: " << to_string(detObj.back().confidence) << endl;
		cout << leftbottom << " " << righttop << endl;
	}
	imshow("Detected", src);
	waitKey(0);
	return 0;
}