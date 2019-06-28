#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image         | <none> | image to process        }"
"{ m  model         |        | path to model weights   }"
"{ c  config        |        | path to model config    }"
"{ w  width         |        | net input width         }"
"{ h  height        |        | net input height        }"
"{ h ? help usage   |        | print help message      }"
"{ b  backend       |   0    | Choose one of computation backends: "
                               "0: automatically (by default), "
                               "1: Halide language, "
                               "2: Intel's Deep Learning Inference Engine, "
                               "3: OpenCV implementation }"
"{ t  target        |   0    | Choose one of target computation devices: "
                               "0: CPU target (by default), "
                               "1: OpenCL, "
                               "2: OpenCL fp16 (half-float)";

struct NetParams
{
	float scale = 1.0;
	Scalar mean = Scalar(0.0, 0.0, 0.0, 0.0);
	bool swapRB = false;
	int inpWidth = 227;
	int inpHeight = 227;
	String framework = "";
	String model = "../../data/squeezenet1.1.caffemodel";
	String config = "../../data/squeezenet1.1.prototxt";
	string classesNames = "../../data/squeezenet1.1.labels";
	int backendId = 0;
	int targetId = 0;
};



int main(int argc, char** argv)
{
	// Input parameters
	string imageName = "../../data/unn_neuromobile.jpg";
	NetParams p;


	// Loading image
	Mat source = imread(imageName, 1);

	// Loaging Net
	DnnClassificator* dnnclas = new DnnClassificator(p.model, p.config, p.inpWidth, p.inpHeight);
	dnnclas->loadClassesNames(p.classesNames);

	// Run Network
	Mat prob = dnnclas->Classify(source);

	// Choose best class
	Point classIdPoint;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;

	// Print class name
	cout << "Predicted class: " << dnnclas->classesNames[classId] << endl;

	waitKey(0); // Wait for a keystroke in the window
	delete dnnclas;
	return 0;
}