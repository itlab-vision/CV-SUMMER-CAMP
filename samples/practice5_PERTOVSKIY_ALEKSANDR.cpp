#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <random>
#include <iostream>


#include "detector.h"
#include "tracking_by_matching.hpp"
#include "label_parser.h"
#include "classificator.h"

using namespace std;
using namespace cv;
using namespace cv::tbm;

static const char* keys =
{ "{video_name       | | video name                       }"
"{start_frame      |0| Start frame                      }"
"{frame_step       |1| Frame step                       }"
"{detector_model   | | Path to detector's Caffe model   }"
"{detector_weights | | Path to detector's Caffe weights }"
"{desired_classes |-1| The desired classes that should be tracked }"
"{detector_labels_path  | | Path to detector's labels        }"
"{classificator_labels_path | | Path to classificator's labels        }"
"{classificator_model  | | Path to classificator's model        }"
"{classificator_weights | | Path to classificator's weights        }"
};



static void help()
{
	cout << "\nThis example shows the functionality of \"Tracking-by-Matching\" approach:"
		" detector is used to detect objects on frames, \n"
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



struct params {
	String model_path;
	String config_path;
	String label_path;
	Scalar mean;
	double scale;
	bool swapRB;
	int width;
	int height;
};

const std::map<String, Scalar> colours = { { "cat", Scalar(0, 36, 255) },
{ "dog", Scalar(0, 255, 0) } };

vector<int> parseClassesString(String inputClasses) {
	vector<int> classes;
	int i = 0;
	while (inputClasses[i] == ' ') { i++; }
	inputClasses.erase(inputClasses.begin(), inputClasses.begin() + i);
	int j = 0;
	int prev = 0;
	int inpSize = inputClasses.size();
	while (j < inpSize) {
		if (inputClasses[j] == ' ') {
			String subStr = inputClasses.substr(prev, j);
			classes.push_back(std::stoi(subStr));
			prev = j + 1;
		}
		j++;
	}
	String subStr = inputClasses.substr(prev, inpSize);
	classes.push_back(std::stoi(subStr));
	prev = j + 1;
	return classes;
}

cv::Ptr<ITrackerByMatching> createTrackerByMatchingWithFastDescriptor();


cv::Ptr<ITrackerByMatching>
createTrackerByMatchingWithFastDescriptor() {
	cv::tbm::TrackerParams params;

	cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatching(params);

	std::shared_ptr<IImageDescriptor> descriptor_fast =
		std::make_shared<ResizedImageDescriptor>(
			cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
	std::shared_ptr<IDescriptorDistance> distance_fast =
		std::make_shared<MatchTemplateDistance>();

	tracker->setDescriptorFast(descriptor_fast);
	tracker->setDistanceFast(distance_fast);

	return tracker;
}


// return classname and confidence
std::pair<String, double> classifyDog(const Mat& image, params squez_params) {
	std::map<int, String> classes = initializeClasses(squez_params.label_path);
	DnnClassificator squeez_cl(squez_params.model_path,
		squez_params.config_path,
		squez_params.label_path,
		squez_params.width,
		squez_params.height,
		squez_params.mean,
		squez_params.swapRB);
	Mat probReshaped = squeez_cl.Classify(image);
	double confidence;
	Point classPoint;

	minMaxLoc(probReshaped, 0, &confidence, 0, &classPoint);
	return std::make_pair(classes[classPoint.x], confidence);

}

int main(int argc, char** argv) {

	CommandLineParser parser(argc, argv, keys);
	cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();

	String video_name = parser.get<String>("video_name");
	int start_frame = parser.get<int>("start_frame");
	int frame_step = parser.get<int>("frame_step");
	String desired_classes = parser.get<String>("desired_classes");

	params mobilenet_params = {
		parser.get<String>("detector_model"),
		parser.get<String>("detector_weights"),
		parser.get<String>("detector_labels_path"),
		Scalar(127.5, 127.5, 127.5), // mean
		0.007843, // scale
		false, // swapRB
		300, // width
		300 // height
	};

	params squeeze_params = {
		parser.get<String>("classificator_model"),
		parser.get<String>("classificator_weights"),
		parser.get<String>("classificator_labels_path"),
		Scalar(104.0, 117.0, 123.0), // mean
		1, // scale
		false, // swapRB
		227, // width
		227 // height
	};
	



	vector<int> classes = parseClassesString(desired_classes);

	//if (video_name.empty() || mobilenet_params.model_path.empty() || mobilenet_params.config_path.empty())
	//{
	//	help();
	//	return -1;
	//}

	//open the capture
	VideoCapture cap;
	cap.open(video_name);
	cap.set(CAP_PROP_POS_FRAMES, start_frame);

	if (!cap.isOpened())
	{
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		parser.printMessage();
		return -1;
	}


	DnnDetector detector(mobilenet_params.model_path,
		mobilenet_params.config_path, 
		mobilenet_params.label_path,
		mobilenet_params.width,
		mobilenet_params.height, 
		mobilenet_params.scale, 
		mobilenet_params.mean,
		mobilenet_params.swapRB);
	

	Mat frame;
	namedWindow("Tracking by Matching", 1);

	int frame_counter = -1;
	int64 time_total = 0;
	bool paused = false;
	std::vector<String> already_save;
	for (;; )
	{
		if (paused)
		{
			char c = (char)waitKey(30);
			if (c == 'p')
				paused = !paused;
			if (c == 'q')
				break;
			continue;
		}

		cap >> frame;
		if (frame.empty()) {
			break;
		}
		frame_counter++;
		if (frame_counter < start_frame)
			continue;
		if (frame_counter % frame_step != 0)
			continue;


		int64 frame_time = getTickCount();

		TrackedObjects detections = detector.Detect(frame, frame_counter);



		// timestamp in milliseconds
		uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / 30 * frame_counter);
		tracker->process(frame, detections, cur_timestamp);

		frame_time = getTickCount() - frame_time;
		time_total += frame_time;

		// Drawing colored "worms" (tracks).
		frame = tracker->drawActiveTracks(frame);

		for (const auto &detection : detections) {
			cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
		}

		// Drawing tracked detections only by RED color and print ID and detection
		// confidence level.



		for (const auto &detection : tracker->trackedDetections()) {
			std::pair<cv::String, double> dogClass;
			if (detection.class_id == 12 && detection.confidence > 0.5) {
				Mat subframe = frame(detection.rect);
				dogClass = classifyDog(frame, squeeze_params);
				if (std::find(already_save.begin(), already_save.begin(), dogClass.first) != already_save.end()) {
					imwrite("../" + dogClass.first + ".jpg", subframe);
					already_save.push_back(dogClass.first);
				}
			}else {
				continue;
			}

		}

		imshow("Tracking by Matching", frame);

		char c = (char)waitKey(2);
		if (c == 'q')
			break;
		if (c == 'p')
			paused = !paused;
	}

	double s = frame_counter / (time_total / getTickFrequency());
	printf("FPS: %f\n", s);

	return 0;
}
