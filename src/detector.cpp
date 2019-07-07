#include "detector.h"
#include <conio.h>

DnnDetector::DnnDetector(const std::string &pathToModel, const std::string &pathToConfig, const std::string &pathToLabel,
	const std::int32_t &inputWidth, const std::int32_t &inputHeight, const std::double_t scale, const cv::Scalar &mean, const bool &swapRB)
{
	m_pathToModel = pathToModel;
	m_pathToConfig = pathToConfig;
	m_pathToLabel = pathToLabel;

	m_width = inputWidth;
	m_height = inputHeight;

	m_scale = scale;
	m_mean = mean;
	m_swapRB = swapRB;

	m_net = cv::dnn::readNet(pathToModel, pathToConfig);

	m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	std::cout << "model: " << m_pathToModel << std::endl;
	std::cout << "config: " << m_pathToConfig << std::endl;
	std::cout << "label: " << m_pathToLabel << std::endl;

	std::cout << "W: " << m_width << ", H: " << m_height << std::endl;
	std::cout << "Scale: " << m_scale << std::endl;
	std::cout << "Mean: " << m_mean << std::endl;
	std::cout << "SwapRB: " << swapRB << std::endl;
}


vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	cv::Mat inputTensor;

	cv::dnn::blobFromImage(image, inputTensor, m_scale, cv::Size(m_width, m_height), m_mean, m_swapRB);

	m_net.setInput(inputTensor);
	prob = m_net.forward().reshape(1, 100);

	for (std::uint32_t i = 0; i < prob.rows; i++)
	{
		DetectedObject object;

		object.uuid =	prob.at<std::float_t>(i, 1);
		std::cout << "Confidence: " << i << prob.at<std::float_t>(i, 2) << std::endl;

		object.Left =	prob.at<std::float_t>(i, 3);
		object.Bottom = prob.at<std::float_t>(i, 4);
		object.Right =	prob.at<std::float_t>(i, 5);
		object.Top =	prob.at<std::float_t>(i, 6);

		m_obects.push_back(object);
	}

	std::uint32_t counter = 0;
	for (auto &obj : m_obects)
	{
		std::cout << "Obj " << counter << ":\t";

		std::cout << "classId: " << obj.uuid << ",\t";

		std::cout << "L: " << obj.Left << ",\t";
		std::cout << "B: " << obj.Bottom << ",\t";
		std::cout << "R: " << obj.Right << ",\t";
		std::cout << "T: " << obj.Top << std::endl;

		counter++;
	}

	return m_obects;
}