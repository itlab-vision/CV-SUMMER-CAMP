#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0 {}
};



class DnnDetector : public Detector
{
public:
	DnnDetector(const std::string &pathToModel, const std::string &pathToConfig, const std::string &pathToLabel,
		const std::int32_t &inputWidth, const std::int32_t &inputHeight, const std::double_t scale, const cv::Scalar &mean = cv::Scalar(0, 0, 0, 0), const bool &swapRB = false);

	virtual vector<DetectedObject> Detect(Mat image);

private:
	std::string m_pathToModel, m_pathToConfig, m_pathToLabel;

	std::int32_t m_width, m_height;

	std::double_t m_scale;
	cv::Scalar m_mean;
	bool m_swapRB;

	cv::dnn::Net m_net;
	cv::Mat prob;

	std::vector<DetectedObject> m_obects;

};