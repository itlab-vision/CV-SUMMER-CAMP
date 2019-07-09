#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "tracking_by_matching.hpp"
#include <fstream>
#include <string>

#include "detector.h"


using namespace std;
using namespace cv;
using namespace cv::tbm;


static const char* keys = {
	"{video_name       |  | video name                               }"
	"{start_frame      | 0| Start frame                              }"
	"{frame_step       | 1| Frame step                               }"
	"{detector_model   |  | Path to detector's Caffe model           }"
	"{detector_weights |  | Path to detector's Caffe weights         }"
	"{desired_class_id |-1| The desired class that should be tracked }"
};

static void help()
{
	cout << "\nThis example shows the functionality of \"Tracking-by-Matching\" approach:"
		"detector is used to detect objects on frames, \n"
		"matching is used to find correspondences between new detections and tracked objects.\n"
		"Detection is made by DNN detection network every `--frame_step` frame.\n"
		"Point a .prototxt file of the network as the parameter `--detector_model`, and a .caffemodel file"
		" as the parameter `--detector_weights`.\n"
		"(As an example of such detection network is a popular MobileNet_SSD network trained on VOC dataset.)\n"
		"If `--desired_class_id` parameter is set, the detection result is filtered by class id,"
		" returned by the detection network.\n"
		"(That is, if a detection net was trained on VOC dataset, then to track pedestrians point --desired_class_id=15)\n"
		"Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
		"Call:\n"
		"./example_tracking_tracking_by_matching --video_name=<video_name> --detector_model=<detector_model_path> --detector_weights=<detector_weights_path> \\\n"
		"                                       [--start_frame=<start_frame>] \\\n"
		"                                       [--frame_step=<frame_step>] \\\n"
		"                                       [--desired_class_id=<desired_class_id>]\n"
		<< endl;

	cout << "\n\nHot keys: \n"
		"\tq - quit the program\n"
		"\tp - pause/resume video\n";
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


int main(int argc, char** argv) {
	CommandLineParser parser(argc, argv, keys);

	String video_name = parser.get<String>("video_name");
	String detector_model = parser.get<String>("detector_model");
	String detector_weights = parser.get<String>("detector_weights");
	int start_frame = parser.get<int>("start_frame");
	int frame_step = parser.get<int>("frame_step");
	int desired_class_id = parser.get<int>("desired_class_id");

	vector<String> labels = readLabel("C:/practice/CV-SUMMER-CAMP/data/model/labels.txt");

	if (video_name.empty() || detector_model.empty() || detector_weights.empty()) {
		help();
		return -1;
	}

	VideoCapture cap(video_name);
	cap.set(CAP_PROP_POS_FRAMES, start_frame);

	if (!cap.isOpened()) {
		cout << "***Could not initialize capturing...***\n";
		return -1;
	}

	Mat frame;
	namedWindow("Tracking");
	DnnDetector detector(detector_model, detector_weights, desired_class_id);
	for (cap >> frame; !frame.empty(); cap >> frame) {
		TrackedObjects detections = detector.Detect(frame);

		for (const auto &detection : detections) {
			rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
			if (detection.object_id < labels.size()) {
				putText(frame, labels[detection.object_id], { detection.rect.x, detection.rect.y + detection.rect.height }, FONT_HERSHEY_PLAIN, 2, { 0, 255, 0 }, 2);
			}
		}

		imshow("Tracking", frame);
		waitKey(1);
	}

	system("pause");
	return 0;
}
