#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "tracking_by_matching.hpp"
#include <fstream>
#include <string>
#include <map>

#include "detector.h"
#include "classificator.h"


using namespace std;
using namespace cv;
using namespace cv::tbm;


struct DetectedFrame {
	int id;
	Rect rect;
	Mat frame;
};


static const char* keys = {
	"{video_name       |  | video name                               }"
	"{start_frame      | 0| Start frame                              }"
	"{frame_step       | 1| Frame step                               }"
	"{detector_model   |  | Path to detector's Caffe model           }"
	"{detector_weights |  | Path to detector's Caffe weights         }"
	"{desired_class_id |-1| The desired class that should be tracked }"
};


template<typename Iterator, typename Elem>
bool contains(Iterator start, const Iterator& end, Elem elem) {
	while (start != end) {
		if (*start == elem) {
			return true;
		}
		++start;
	}
	return false;
}


DetectedFrame findMax(const TrackedObjects& objects, const Mat& frame) {
	double confidence = objects[0].confidence;
	size_t index = 0;
	for (size_t i = 1; i < objects.size(); ++i) {
		if (objects[i].confidence > confidence) {
			confidence = objects[i].confidence;
			index = i;
		}
	}
	return { objects[index].object_id, objects[index].rect, frame };
}


DetectedFrame findMax(const Mat& mat, const Rect& rect, const Mat& frame) {
	Point point;
	minMaxLoc(mat, nullptr, nullptr, nullptr, &point);
	return { point.x, rect, frame };
}


std::vector<std::string> readLabel(const String& path) {
	std::ifstream is(path);
	std::string temp;
	std::vector<std::string> result;
	while (std::getline(is, temp)) {
		if (temp.size()) {
			result.push_back(temp);
		}
	}
	return result;
}


std::vector<cv::Mat> toArray(VideoCapture& cap) {
	Mat frame;
	std::vector<cv::Mat> result;
	for (cap >> frame; !frame.empty(); cap >> frame) {
		result.push_back(frame);
	}
	return result;
}


std::vector<DetectedFrame> findAll(DnnDetector& detector, const std::vector<cv::Mat>& frames) {
	std::vector<DetectedFrame> result;
	for (size_t i = 0; i < frames.size(); ++i) {
		auto detected = detector.Detect(frames[i]);
		if (!detected.empty()) {
			result.push_back(findMax(detector.Detect(frames[i]), frames[i]));
		}
	}
	return result;
}


std::vector<DetectedFrame> findUnique(DnnClassificator& classificator, const std::vector<DetectedFrame>& frames) {
	std::vector<int> keys;
	std::vector<DetectedFrame> result;
	for (const auto& frame : frames) {
		Mat detected_array = classificator.Classify(frame.frame);
		DetectedFrame detected = (findMax(detected_array, frame.rect, frame.frame));
		if (!contains(keys.begin(), keys.end(), detected.id)) {
			result.push_back(detected);
			keys.push_back(detected.id);
		}
	}
	return result;
}


int main(int argc, char** argv) {
	CommandLineParser parser(argc, argv, keys);

	String video_name = parser.get<String>("video_name");
	String detector_model = parser.get<String>("detector_model");
	String detector_weights = parser.get<String>("detector_weights");
	int start_frame = parser.get<int>("start_frame");
	int frame_step = parser.get<int>("frame_step");
	int desired_class_id = parser.get<int>("desired_class_id");

	vector<String> labels = readLabel("E:/Projects/practice/CV-SUMMER-CAMP/data/model/squeezenet1.1.labels");
	std::string outputPath = "E:/Projects/practice/CV-SUMMER-CAMP/data/output/";

	VideoCapture cap(video_name);
	cap.set(CAP_PROP_POS_FRAMES, start_frame);

	DnnDetector dogDetector(detector_model, detector_weights, desired_class_id);
	DnnClassificator breedDetector("E:/Projects/practice/CV-SUMMER-CAMP/data/model/squeezenet1.1.caffemodel",
		"E:/Projects/practice/CV-SUMMER-CAMP/data/model/squeezenet1.1.prototxt", 227, 227, 0, { 104.f, 117.f, 123.f });

	auto dogs = findAll(dogDetector, toArray(cap));
	auto breed = findUnique(breedDetector, dogs);

	std::ofstream file;
	file.open(outputPath + "breeds.txt");
	for (size_t i = 0; i < breed.size(); ++i) {
		std::string name = labels[breed[i].id];
		file << name << std::endl;
		imwrite(outputPath + name + ".jpg", breed[i].frame(breed[i].rect));
	}
	file.close();

	system("pause");
	return 0;
}