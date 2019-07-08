#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/dnn.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  widht                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";

Point p1, p2;
bool lup, ldown;
Mat img;
static void mouse_do_nothing(int event, int x, int y, int, void*)
{
	return;
}
static void selectROI(int event,int x,int y,int,void*)
{
	if (event == EVENT_RBUTTONDOWN)
	{
		return;
	}
	if (event == EVENT_LBUTTONDOWN)
	{
		ldown = true;
		p1.x = x;
		p1.y = y;
	}
	if (event == EVENT_LBUTTONUP)
	{
		if (abs(x - p1.x) > 10 && abs(y - p1.y) > 10)
		{
			lup = true;
			p2.x = x;
			p2.y = y;
		}
		else lup = true;
	}
	
	if (ldown == true && lup == false)
	{
		Point p(x, y);
		Mat locImg = img.clone();
		rectangle(locImg, p1, p, Scalar(0, 255, 0));
		namedWindow("Image");
		imshow("Image", locImg);
	}
	if (ldown == true && lup == true)
	{
		Rect rect;
		rect.width = abs(p1.x - p2.x);
		rect.height = abs(p1.y - p2.y);
		rect.x = min(p1.x, p2.x);
		rect.y = min(p1.y, p2.y);
		img = Mat(img, rect);
		ldown = false;
		lup = false;
	}
}

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
	int widht = parser.get<int>("widht");
	int heigth = parser.get<int>("heigth");
	String model_path(parser.get<String>("model_path"));
	String config_path(parser.get<String>("config_path"));
	String label_path(parser.get<String>("label_path"));
	bool swap = parser.get<bool>("swap");
	Scalar mean = parser.get<Scalar>("mean");

	string msg[4] = { "Highlight the ROI on the image with ",
						 "the left mouse button.Then press any key.",
						  "Right - click to classify the entire image, ",
						  "or wait 30 seconds." };
	Mat helpmsg = Mat(Size(400, 80),1);
	for (int i=0;i<4;i++)
	{
		putText(helpmsg, msg[i], Point(10,20+i*15), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 255, 255));
	};
	

	img = imread(imgName);
	namedWindow("Image");
	imshow("Image", img);
	imshow("help", helpmsg);
	setMouseCallback("Image", selectROI);
	waitKey(30000);
	setMouseCallback("Image", mouse_do_nothing);
	//Image classification
	DnnClassificator classificator(model_path, config_path, label_path, widht, heigth, mean, swap);
	Mat res = classificator.Classify(img);
	
	//Show result


	Point classIdPoint[3] = { (0,0),(0,0), (0,0) };
	double confidence[3] = { 0,0,0 };
	int classId[3] = { 0,0,0 };
	for (int i = 0; i < 5; i++)
	{
		minMaxLoc(res, 0, &confidence[i], 0, &classIdPoint[i]);
		res.at<float>(0, classIdPoint[i].x) = 0;
		classId[i] = classIdPoint[i].x;
	}



	ifstream labels(label_path);
	int i = 1;
	string label[3];
	for (int s = 0; s < 3; s++)
	{
		i = 0;
		labels.seekg(0, ifstream::beg);
		while (labels.is_open())
		{
			getline(labels, label[s]);
			if (i == classId[s])
			{
				break;
			}
			i++;
		}
	}

	string result[3];
	Scalar color[3] = { {190,0,0},{0,190,0},{0,0,190} };

	for (int i = 0; i < 3; i++)
	{
		result[i] = to_string(i+1) + ":" + label[i] + "(" + to_string(classId[i]) + ") Confidence" + to_string(confidence[i]);
		putText(img, result[i], Point(5, img.rows-10- i * 20), FONT_HERSHEY_DUPLEX, 0.4, color[i]);
	}


	imshow("Image", img);
	waitKey();
	return 0;
}



//-i="C:\SummerCampRepos\CV-SUMMER-CAMP\data\unn_neuromobile.jpg" -w="224" -h="224" -model_path="C:\SummerCampRepos\CV-SUMMER-CAMP\data\squeezenet1.1.caffemodel" -config_path="C:\SummerCampRepos\CV-SUMMER-CAMP\data\squeezenet1.1.prototxt" -label_path="C:\SummerCampRepos\CV-SUMMER-CAMP\data\squeezenet1.1.labels" -mean = "103.94 116.78 123.68" -swap ="TRUE"