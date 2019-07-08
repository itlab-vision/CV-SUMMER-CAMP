#include "detector.h"
#include <conio.h>

DnnDetector::DnnDetector(std::string pathToModel, std::string pathToConfig, std::string pathToLabel,
	std::int32_t inputWidth, std::int32_t inputHeight, std::double_t scale, cv::Scalar mean, bool swapRB)
{
	m_pathToModel = pathToModel;
	m_pathToConfig = pathToConfig;
	m_pathToLabel = pathToLabel;

	m_width = inputWidth;
	m_height = inputHeight;

	m_scale =  1.0 / scale;
	m_mean = mean;
	m_swapRB = swapRB;

	m_net = cv::dnn::readNet(pathToModel, pathToConfig);

	m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}


vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	std::vector<DetectedObject> detObjects;
	cv::Mat inputTensor;

	cv::dnn::blobFromImage(image, inputTensor, m_scale, cv::Size(m_width, m_height), m_mean, m_swapRB);

	m_net.setInput(inputTensor);
	prob = m_net.forward().reshape(1, 1);
	prob.reshape(1, prob.cols / 7).copyTo(prob);
	
	for (std::uint32_t i = 0; i < prob.rows; i++)
	{
		DetectedObject object;

		object.classId =	prob.at<std::float_t>(i, 1);
		object.confidence = prob.at<std::float_t>(i, 2);

		object.Left =	prob.at<std::float_t>(i, 3) * image.size().width;
		object.Bottom = prob.at<std::float_t>(i, 4) * image.size().height;
		object.Right =	prob.at<std::float_t>(i, 5) * image.size().width;
		object.Top =	prob.at<std::float_t>(i, 6) * image.size().height;

		detObjects.push_back(object);
	}

	return detObjects;
}



void  DnnDetector::showParams()
{
	std::cout << "model: " << m_pathToModel << std::endl;
	std::cout << "config: " << m_pathToConfig << std::endl;
	std::cout << "label: " << m_pathToLabel << std::endl;

	std::cout << "W: " << m_width << ", H: " << m_height << std::endl;
	std::cout << "Scale: " << m_scale << std::endl;
	std::cout << "Mean: " << m_mean << std::endl;
	std::cout << "SwapRB: " << m_swapRB << std::endl;
}