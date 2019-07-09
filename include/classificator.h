#pragma once
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Classificator
{
public:
    vector<string> classesNames;
    virtual Mat Classify(Mat image) = 0 {}
};



class DnnClassificator : public Classificator
{
public:
	DnnClassificator(const std::string &pathToModel, const std::string &pathToConfig, const std::string &pathToLabel, 
		const int &inputWidth, const int &inputHeight, const cv::Scalar &mean = cv::Scalar(0, 0, 0, 0), const bool &swapRB = false);

	Mat Classify(Mat image);

	void showTheBestClass();
	void showTheBestClasses(const std::uint32_t &numberOfClasses);

private:
	std::string m_pathToModel, m_pathToConfig, m_pathToLabel;
	
	int m_width, m_height;
	cv::Scalar m_mean;
	bool m_swapRB;

	cv::dnn::Net m_net;
	cv::Mat prob;

};