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
    String model_path(parser.get<String>("model_path"));
    String config_path(parser.get<String>("config_path"));
    String path_label(parser.get<String>("label_path"));
    int width(parser.get<int>("width"));
    int height(parser.get<int>("heigth"));
    Scalar mean(parser.get<Scalar>("mean"));
    int swapRB(parser.get<int>("swap"));

    DnnClassificator dnn(model_path, config_path, 
      path_label, width , height, mean, swapRB);

    Mat image = imread(imgName);
    Mat prob;
	//Image classification
    
    prob = dnn.Classify(image);
	
	//Show result
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);

    int classId = classIdPoint.x;

    std::cout << "Class: " << classId << '\n';
    std::cout << "Confidence: " << confidence << '\n';

    std::string name;

    std::ifstream in("../../CV-SUMMER-CAMP/data/squeezenet1.1.labels"); 
    int count = 0;
    if (in.is_open())
    {
        while (getline(in, name))
        {
            if (count == classId)
            {
                break;
            }
            count++;

        }
    }
    in.close();



    Mat res = image;
    putText(res, "Class: "+to_string(classId), Point(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(200, 200, 250));
    putText(res, "Confidence: " + to_string(confidence), Point(30, 60), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(200, 200, 250));
    putText(res, name, Point(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(200, 200, 250));


    imshow("image", res);
    waitKey();
	return 0;
}
