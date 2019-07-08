/*#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

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

int main(int argc, char** argv)
{
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
	// Load image and init parameters
	String imgName(parser.get<String>("image"));
	
	Mat image = imread(imgName);
	String modelPath = parser.get<String>("model_path");
	String configPath = parser.get<String>("config_path");
	String labelsPath = parser.get<string>("label_path");
	int width = parser.get<int>("width");
	int height = parser.get<int>("heigth");
	Scalar mean = parser.get<Scalar>("mean");
	bool swapRB = parser.get<bool>("swap");
	
	
	//Image classification
	Classificator* cl = new DnnClassificator(modelPath, configPath, labelsPath, width, height, mean, swapRB);
	Mat prob = cl->Classify(image);
	
	
	//Show result
	Point classIdPoint;
	double confidence;
	minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	cout << "Class: " << classId << '\n';
	cout << "Confidence: " << confidence << '\n';
	return 0;
	
	
}*/



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
"{ label_path     |        |                  }";
/*"{ width		  |        |                  }"
"{ height	      |        |                  }"
"{ mean		      |        |                  }"
"{ swap		      |        |                  }"*/

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
	/*int width = parser.get<int>("width");
	int height = parser.get<int>("height");
	Scalar mean = parser.get<Scalar>("mean");
	bool swapRB = parser.get<bool>("swap");*/
	modelPath = "C:\\Users\\world\\CV-SUMMER-CAMP\\data\\new\\object_detection\\common\\mobilenet-ssd\\caffe\\mobilenet-ssd.caffemodel";
	configPath = "C:\\Users\\world\\CV-SUMMER-CAMP\\data\\new\\object_detection\\common\\mobilenet-ssd\\caffe\\mobilenet-ssd.prototxt";
	labelsPath = "C:\\Users\\world\\CV-SUMMER-CAMP\\data\\mobilenet-ssd.labels";
	imgName = "C:\\Users\\world\\CV-SUMMER-CAMP\\data\\unn_neuromobile.jpg";
	Mat image = cv::imread(imgName);
	Detector* det = new DnnDetector(modelPath, configPath, labelsPath);
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