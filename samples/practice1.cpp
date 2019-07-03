#include <iostream>
#include <string>
#include <ctime>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>

#include "filter.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image         | <none> | image to process        }"
"{ w  width         | <none> | width for image resize  }"
"{ h  height        | <none> | height for image resize }"
"{ c  count         | <none> | count for Piytnashki    }"
"{ q ? help usage   | <none> | print help message      }";

int main(int argc, char** argv){
    // Process input arguments
    CommandLineParser parser(argc, argv, cmdOptions);
    parser.about(cmdAbout);     

    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    if (!parser.check()){    
        parser.printErrors();
        return 0;
    }
    
    // Load image
	//String imgName(parser.get<String>("image")); //path
	cv::Mat image;// = cv::imread(imgName);
	/*
	int width = (parser.get<int>("width")); //100
	int height = (parser.get<int>("height")); //100
    
    // Filter image
	Filter* grayImage = new GrayFilter();
	image = grayImage->ProcessImage(image);
	Filter* resizeImage = new ResizeFilter(width, height);
	image = resizeImage->ProcessImage(image);

	//Pyatnashka
	Mat res = image;
	int count = (parser.get<int>("count")); //4
	count = 4;
	int* randomMas = new int [count*count];
	srand(time(0));
	for (int i = 0; i < count*count; i++){ //заполнение массива случайными неповторяющимиеся числами 
		do{
			randomMas[i] = 1 + rand() % (count*count);
		} while(prov(i, randomMas, randomMas[i]));
	}
	int widhtPyat = image.cols/count;
	int heightPyat = image.rows/count;
	Mat** Pyat = new Mat*[count];
	for (int i = 0; i < count; i++) Pyat[i] = new Mat[count];
	//for (int i = 0; i < count*count; i++) {
	Mat tmp = image(Rect(widhtPyat, heightPyat, widhtPyat, heightPyat));
	Mat tmp1 = image(Rect(0, 0, widhtPyat, heightPyat));
		//cvSetImageROI(image, Rect(0, 0, widhtPyat, heightPyat));
	tmp1.setTo(tmp);*/

	//}
	//Realtime

	cv::VideoCapture cap(0);
	cv::namedWindow("My image", cv::WINDOW_NORMAL);
	for(;;){
		cap >> image;
		cv::imshow("My image", image);

	}

    // Show image

	//cv::namedWindow("My image", cv::WINDOW_NORMAL);
	cv::waitKey(0);
    
    
    
    
    return 0;
}
