#include <string>
#include <iostream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <ctime> 
#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
	"{ i image        |        | image to process         }"
    "{ model_path     |        |                  }"
    "{ config_path    |        |                  }"
    "{ label_path     |        |                  }"
    "{ h ? help usage |        | print help message       }";

Mat statistic(Mat img, int i, std::chrono::duration<double> time) {
	string txt = to_string(i) + "/" + to_string(time.count());
	cv::rectangle(img, Point(img.cols, img.rows), Point(img.cols-30, img.rows-10), (255, 255, 255), FILLED);
	putText(img, txt, Point(img.cols - 28, img.rows - 2), FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0));
	return img;
}


int main(int argc, const char** argv) {
	// Parse command line arguments.
	CommandLineParser parser(argc, argv, cmdOptions);
	parser.about(cmdAbout);

	// If help option is given, print help message and exit.
	if (parser.get<bool>("help")) {
	parser.printMessage();
	return 0;
	}

	VideoCapture cap;
	cap.open(0);
	VideoWriter *writer = new VideoWriter();

	// Do something cool.

	String imgName(parser.get<String>("image")); //path
	string model(parser.get<string>("model_path"));
	string config(parser.get<string>("config_path"));
	string labels(parser.get<string>("label_path"));

	model = "C:/CV-Intel/CV-SUMMER-CAMP/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel";
	config = "C:/CV-Intel/CV-SUMMER-CAMP/mobilenet-ssd/caffe/mobilenet-ssd.prototxt";
	imgName = "C:/CV-Intel/CV-SUMMER-CAMP/data/qwe.jpg";
	String label[21] = { "background", "aeroplane", "bicycle", "bird", 
	"boat","bottle", "bus",  "car", "cat", "chair", "cow",
	"diningtable",  "dog", "horse", "motorbike", "person", 
	"pottedplant", "sheep",  "sofa", "train", "tvmonitor" };

	//Mat image = cv::imread(imgName);

	Mat image;
	cap >> image;
	Size *s = new Size((int)image.cols, image.rows);
	writer->open("C:/Users/Admin/Desktop/MyVideo.avi", -1, 30, *s, true);
	auto start = std::chrono::system_clock::now();
	auto end = std::chrono::system_clock::now();
	int count=0;
	// Some computation here
	while (waitKey(1) < 0){
		cap >> image;
		//double rate = cap.get();

		Detector* detect = new DnnDetector(model, config, labels);
		vector<DetectedObject> mat = detect->Detect(image);

		for (int i = 0; i < mat.size(); i++) {
			if (mat[i].confidence > 0.5) {
				if (count == 0) {
					start = std::chrono::system_clock::now();
					count = mat.size();
				}
				string text = label[mat[i].classid] + " : " + to_string(mat[i].confidence);
				cv::rectangle(image, Point(mat[i].Left, mat[i].Top), Point(mat[i].Right, mat[i].Bottom), (255, 255, 255), 1);
				cv::rectangle(image, Point(mat[i].Left, mat[i].Bottom), Point(mat[i].Right, mat[i].Bottom - 20), (255, 255, 255), FILLED);
				putText(image, text, Point(mat[i].Left, mat[i].Bottom-5), FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0));
			}
		}
		if (count == mat.size()) {
			end = std::chrono::system_clock::now();
		} else{
			count = 0;
			start = std::chrono::system_clock::now();
			end = std::chrono::system_clock::now();
		}
		std::chrono::duration<double> t = end - start;
		image = statistic(image, mat.size(),t);

		cv::namedWindow("My image", cv::WINDOW_NORMAL);
		cv::imshow("My image", image);
		*writer << image;
		//break;
	}
	cv::waitKey(0);
	std::cout << "This is empty template sample." << endl;

	return 0;
}