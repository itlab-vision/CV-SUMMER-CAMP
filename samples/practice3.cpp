#include <string>
#include <iostream>
#include "detector.h"

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
"{ i  image                             | <none> | image to process                  }"
"{ m  model_path                           |        | path to model                     }"
"{ c  config_path                          |        | path to model configuration       }"
"{ l  label_path                           |        | path to class labels              }"
"{ h ? help usage                       |        | print help message                }";


int main(int argc, const char** argv) {
	// Parse command line arguments.
	CommandLineParser parser(argc, argv, cmdOptions);
	parser.about(cmdAbout);

	// If help option is given, print help message and exit.
	if (parser.get<bool>("help")) {
		parser.printMessage();
		return 0;
	}

	string imgName(parser.get<string>("image"));
	string model_path(parser.get<string>("model_path"));
	string config_path(parser.get<string>("config_path"));
	string label_path(parser.get<string>("label_path"));
	Mat image = imread(imgName);

	Detector* dnn = new DnnDetector(model_path, config_path, label_path);
	vector<DetectedObject> vec = dnn->Detect(image);
	for (int i = 0; i < vec.size(); i++) 
	{
		Point p1(vec[i].Left, vec[i].Bottom);
		Point p2(vec[i].Right, vec[i].Top);

		Rect rect(p1, p2);
		rectangle(image, rect, Scalar(0, 255, 0), 2);

		string text = "Class: " + vec[i].className + "Score: " + to_string(vec[i].confidence);

		putText(image,text,p1, FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 0), 1, 8);



		cout << "Class: " << vec[i].className << endl;
		cout << "Confidence: " << vec[i].confidence << endl;
		cout << p1 << " " << p2 << endl;
		

	}
	

	imshow("Detected", image);
	waitKey(0);

	delete dnn;
	return 0;
}