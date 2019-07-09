#include <iostream>
#include <fstream>
#include <string>
#include <set>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth for classification   }"
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
	cv::Mat src;
	src = imread(imgName);
//	cv::imshow("image",src);
	//waitKey();

	//Image classification
	string mp = parser.get<string>("model_path");
	string cp = parser.get<string>("config_path");
	string lp = parser.get<string>("label_path");
	int wid = parser.get<int>("width");
	int hei = parser.get<int>("heigth");
	Scalar me = parser.get<Scalar>("mean");
	int sw = parser.get<int>("swap");
	Point classIdPoint;
	double confidence;
	DnnClassificator ds(mp, cp, lp, wid,hei, me, sw);
	Mat res = ds.Classify(src);
	minMaxLoc(res, 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	
	//Show result
	
	/*ifstream ifs("labels.labels");
	int s = 0;
	while (s <= classId || !ifs.eof())s++;
	string buff;
	getline(ifs, buff);*/
	cout << "Class:" << classId<< endl;
	cout << "Confidence:" << confidence << endl;


	return 0;
}
