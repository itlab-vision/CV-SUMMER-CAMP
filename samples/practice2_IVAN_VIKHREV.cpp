#include <iostream>
#include <fstream>
#include <string>
#include <fstream>

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
	String modelPath = parser.get<String>("model_path");
	String configPath = parser.get<String>("config_path");
	String labelsPath = parser.get<string>("label_path");
	int width = parser.get<int>("width");
	int height = parser.get<int>("heigth");
	Scalar mean = parser.get<Scalar>("mean");
	bool swapRB = parser.get<bool>("swap");

	DnnClassificator dcl(modelPath, configPath, labelsPath, width, height, mean, swapRB);/*parser.get<String>("model_path"), parser.get<String>("config_path"),
		parser.get<string>("label_path"), parser.get<int>("width"), parser.get<int>("height"),
		parser.get<Scalar>("mean"), parser.get<int>("swap"));*/

	Mat image = imread(imgName);
	Mat prob;
	Point classIdPoint[5] = { (0,0), (0,0), (0,0), (0,0) ,(0,0) };
	double confidence[5] = { 0,0,0,0,0 };
	int classId[5] = { 0,0,0,0,0 };
	//Image classification
	prob = dcl.Classify(image);
	//Show result
	Mat tmp = prob.reshape(1, 1);
	for (int i = 0; i < 5; i++) {
		minMaxLoc(tmp, 0, &confidence[i], 0, &classIdPoint[i]);
		tmp.at<float>(0, classIdPoint[i].x) = 0;
		classId[i] = classIdPoint[i].x;
	}
	std::string name;
	std::ifstream in("../../CV-SUMMER-CAMP/data/squeezenet1.1.labels");

	for (int i = 0; i < 5; i++) {
		int count = 0;
		in.seekg(0, ios::beg);
		if (in.is_open())
		{
			while (getline(in, name))
			{
				if (count == classId[i])
				{
					break;
				}
				count++;

			}
		}
		string objClass = "Class: " + std::to_string(classId[i]) + " " + name;
		string conf = "Confidence: " + std::to_string(confidence[i]);
		putText(image, objClass, Size(0, 20+i*40), FONT_HERSHEY_COMPLEX_SMALL, 1,
			Scalar(71, 99, 255), 1, 2);
		putText(image, conf, Size(0, 40+i*40), FONT_HERSHEY_COMPLEX_SMALL, 1,
			Scalar(71, 99, 255), 1, 2);
		
	}
	in.close();
	imshow("win", image);
	waitKey(0);
	return 0;
}
