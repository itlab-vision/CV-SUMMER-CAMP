#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "tracking_by_matching.hpp"
#include <fstream>
#include <string>
#include <map>

#include "detector.h"


using namespace std;
using namespace cv;
using namespace cv::tbm;


struct DetectedFrame {
	int id;
	TrackedObjects obj;
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


std::vector<DetectedFrame> findAll(DnnDetector& detector, std::vector<cv::Mat>& frames) {
	std::vector<DetectedFrame> result;
	for (const auto& frame : frames) {
		DetectedFrame temp;

		temp.obj = detector.Detect(frame);
		temp.frame = frame;
		temp.id = temp.obj.front().object_id;
		result.push_back(temp);
	}
	return result;
}


std::vector<DetectedFrame> findUnique(DnnDetector& detector, std::vector<DetectedFrame>& frames) {
	std::vector<int> keys;
	std::vector<DetectedFrame> result;
	for (const auto& frame : frames) {
		DetectedFrame temp;

		temp.obj = detector.Detect(frame.frame);
		if (std::find(keys.begin(), keys.end(), temp.id) != keys.end()) {
			temp.frame = frame.frame;
			temp.id = temp.obj.front().object_id;
			result.push_back(temp);
			keys.push_back(keys.front());
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

	vector<String> labels = readLabel("C:/practice/CV-SUMMER-CAMP/data/model/labels.txt");

	VideoCapture cap(video_name);
	cap.set(CAP_PROP_POS_FRAMES, start_frame);

	std::vector<cv::Mat> video = toArray(cap);

	DnnDetector dogDetector(detector_model, detector_weights, desired_class_id);
	DnnDetector breedDetector("C:/practice/CV-SUMMER-CAMP/data/model/squeezenet1.1.caffemodel", "C:/practice/CV-SUMMER-CAMP/data/model/squeezenet1.1.prototxt");

	auto dogs = findAll(dogDetector, video);
	auto breed = findUnique(breedDetector, dogs);

	String ouputPath = "C:/practice/CV-SUMMER-CAMP/data/output/";
	std::vector<String> breedsLabels = readLabel("C:/practice/CV-SUMMER-CAMP/data/model/squeezenet1.1.labels");
	std::ofstream file;
	file.open(ouputPath + "breeds.txt");
	for (const auto& image : breed) {
		if (file.is_open()) {
			file << image.id << std::endl;
		}
		imwrite(ouputPath + breedsLabels[image.id] + ".png", image.frame(image.obj.front().rect));
	}
	file.close();

	return 0;
}