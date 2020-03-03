#include "classificator.h"


DnnClassificator::DnnClassificator(const std::string &pathToModel, const std::string &pathToConfig, const std::string &pathToLabel,
	const int &inputWidth, const int &inputHeight, const cv::Scalar &mean, const bool &swapRB)
{
	m_pathToModel = pathToModel;
	m_pathToConfig = pathToConfig;
	m_pathToLabel = pathToLabel;

	// Add classes to vector
	std::ifstream in(m_pathToLabel, std::ios::in);
	if (!in.is_open())
	{
		CV_Error(Error::StsError, "File " + m_pathToLabel + " not found");
	}

	std::string line;
	while (std::getline(in, line))
	{
		classesNames.push_back(line);
	}

	m_width = inputWidth;
	m_height = inputHeight;

	m_mean = mean;

	m_swapRB = swapRB;

	m_net = cv::dnn::readNet(pathToModel, pathToConfig);

	m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}


cv::Mat DnnClassificator::Classify(cv::Mat image)
{
	try
	{
		cv::Mat inputTensor;

		cv::dnn::blobFromImage(image, inputTensor, 1.0, cv::Size(m_width, m_height), m_mean, m_swapRB);

		m_net.setInput(inputTensor);
		prob = m_net.forward().reshape(1, 1);
	}
	catch (cv::Exception &e)
	{
		std::cout << e.what() << std::endl;
		return cv::Mat();
	}

	return prob;
}



void DnnClassificator::showTheBestClass()
{
	std::double_t confidence = 0;
	cv::Point classIdPoint;
	minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);
	std::uint32_t classId = classIdPoint.x;

	cv::String bestClass = cv::format("%s(%d): %.4f",
		(classesNames.empty() ? format("Class #%d", classId).c_str()
		 					  : classesNames[classId].c_str())
							  , classId
							  , prob.at<std::float_t>(0, classId));

	std::cout << bestClass << std::endl;
}
void DnnClassificator::showTheBestClasses(const std::uint32_t &numberOfClasses)
{
	std::vector<int> ids;
	cv::sortIdx(prob, ids, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
	ids.erase(ids.begin(), ids.end() - numberOfClasses);

	for (auto &classId : ids)
	{
		cv::String classStr = format("%s(%d): %.4f", 
			(classesNames.empty() ? format("Class #%d", classId).c_str()
								  : classesNames[classId].c_str())
								  , classId
								  , prob.at<std::float_t>(0, classId));

		std::cout << classStr << std::endl;
	}
}