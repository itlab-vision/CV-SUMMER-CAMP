#include <iostream>
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
"{ h  height                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";


std::string readLabel(const String& path, int number) {
	std::ifstream is(path);
	std::string temp;
	std::vector<std::string> result;
	while (std::getline(is, temp)) {
		if (temp.size()) {
			result.push_back(temp);
		}
	}
	return number < result.size() ? result[number] : "Undef";
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
	String imgName(parser.get<String>("i"));
	String ptm(parser.get<String>("model_path"));
	String ptc(parser.get<String>("config_path"));
	String ptl(parser.get<String>("label_path"));

	uint32_t width = parser.get<int>("width");
	uint32_t height = parser.get<int>("height");

	Scalar mean = parser.get<Scalar>("mean");
	bool swapRB = parser.get<bool>("swap");

	//Image classification
	DnnClassificator dc(ptm, ptc, width, height, mean, swapRB);		

	Point location;
	minMaxLoc(dc.Classify(imread(imgName, IMREAD_COLOR)), nullptr, nullptr, nullptr, &location);

	//Show result

	std::cout << "It's " << readLabel(ptl, location.x) << std::endl;

	system("pause");
	return 0;
}
