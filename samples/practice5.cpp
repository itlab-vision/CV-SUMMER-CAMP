#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "tracking_by_matching.hpp"
#include "classificator.h"
#include <iostream>
#include <vector>
#include <memory>


using namespace std;
using namespace cv;
using namespace cv::tbm;

string detector_weights;
string detector_model;
string classification_weights;
string classification_model;
string labels;
vector<string> lab;

static const char* keys = {
"{video_name       | | video name                       }"
"{start_frame      |0| Start frame                      }"
"{frame_step       |1| Frame step                       }"
"{detector_model   | | Path to detector's Caffe model   }"
"{detector_weights | | Path to detector's Caffe weights }"
"{classification_model | | Path to detector's squeezenet1.1 model }"
"{classification_weights | | Path to detector's squeezenet1.1 weights }"
"{desired_class_id |-1| The desired class that should be tracked }"
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

int squareRR(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4) {
	int left = std::max(x1, x3);
	int top = std::min(y2, y4);
	int right = std::min(x2, x4);
	int bottom = std::max(y1, y3);

	int width = right - left;
	int height = top - bottom;

	if ((width < 0) || (height < 0)) return 0;

	return width * height;
}

int square(int x1, int y1, int x2, int y2) {
	return abs(x1 - x2)*abs(y1 - y2);
}

cv::Ptr<ITrackerByMatching> createTrackerByMatchingWithFastDescriptor();

class DnnObjectDetector {
public:
	DnnObjectDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path, int _desired_class_id, vector<string> _v,
		float confidence_threshold = 0.2,
		//the following parameters are default for popular MobileNet_SSD caffe model
		const String& net_input_name = "data",
		const String& net_output_name = "detection_out",
		double net_scalefactor = 0.007843,
		const Size& net_size = Size(300, 300),
		const Scalar& net_mean = Scalar(127.5, 127.5, 127.5),
		bool net_swapRB = false)
		:confidence_threshold(confidence_threshold),
		net_input_name(net_input_name),
		net_output_name(net_output_name),
		net_scalefactor(net_scalefactor),
		net_size(net_size),
		net_mean(net_mean),
		net_swapRB(net_swapRB)
	{
		net = dnn::readNetFromCaffe(net_caffe_model_path, net_caffe_weights_path);
		v = _v;
		desired_class_id = _desired_class_id;
		if (net.empty())
			CV_Error(Error::StsError, "Cannot read Caffe net");
	}
	TrackedObjects detect(const cv::Mat& frame, int frame_idx)
	{
		Mat resized_frame;
		resize(frame, resized_frame, net_size);
		Mat inputBlob = cv::dnn::blobFromImage(resized_frame, net_scalefactor, net_size, net_mean, net_swapRB);

		net.setInput(inputBlob, net_input_name);
		Mat detection = net.forward(net_output_name);
		Mat detection_as_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		TrackedObjects res;
		for (int i = 0; i < detection_as_mat.rows; i++)
		{
			bool b = true;
			float cur_confidence = detection_as_mat.at<float>(i, 2);
			int cur_class_id = static_cast<int>(detection_as_mat.at<float>(i, 1));
			int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * frame.cols);
			int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * frame.rows);
			int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * frame.cols);
			int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * frame.rows);

			Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));

			if (cur_confidence < confidence_threshold)
				continue;
			if ((desired_class_id >= 0) && (cur_class_id != desired_class_id))
				continue;

			//clipping by frame size
			cur_rect = cur_rect & Rect(Point(), frame.size());
			if (cur_rect.empty())
				continue;
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Mat image(frame, cur_rect);
			Classificator* dnnClassificator = new DnnClassificator(classification_model, classification_weights, labels, 1.0, 227, 227, { 104.0, 117.0, 123.0 }, false);
			Mat prob = dnnClassificator->Classify(image);
			Point classIdPoint;
			double confidence;
			minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
			int classId = classIdPoint.x;

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			string class_st = lab[classId -1];

			TrackedObject cur_obj(cur_rect, cur_confidence, frame_idx, -1, cur_class_id, cur_class_id, class_st);


			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/*
			if ((label[cur_class_id] == "dog") && (cur_obj.saved == false) && (cur_obj.object_id != -1)) {
				string path = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/data/" + std::to_string(cur_obj.object_id) + "_" + lab[classId] + ".jpg";
				Mat res_mat(frame, cur_rect);
				InputArray res(res_mat);
				imwrite(path, res);
				cur_obj.saved = true;
			}*/
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			for (int i = 0; i < v.size(); i++) {
				if (v[i] == label[cur_class_id]) {
					b = false;
				}
			}
			for (int j = 0; j < i; j++) {
				int S = squareRR(x_left, y_bottom, x_right, y_top,
					static_cast<int>(detection_as_mat.at<float>(j, 3) * frame.cols),
					static_cast<int>(detection_as_mat.at<float>(j, 4) * frame.rows),
					static_cast<int>(detection_as_mat.at<float>(j, 5) * frame.cols),
					static_cast<int>(detection_as_mat.at<float>(j, 6) * frame.rows));

				if (S > 0) {
					int S1 = square(x_left, y_bottom, x_right, y_top);
					int S2 = square(static_cast<int>(detection_as_mat.at<float>(j, 3) * frame.cols),
						static_cast<int>(detection_as_mat.at<float>(j, 4) * frame.rows),
						static_cast<int>(detection_as_mat.at<float>(j, 5) * frame.cols),
						static_cast<int>(detection_as_mat.at<float>(j, 6) * frame.rows));
					float obs = (S1 + S2 - S);
					float f = (S / obs);
					if (f > 0.9) {
						if (cur_class_id < static_cast<int>(detection_as_mat.at<float>(j, 1))) {
							b = false;
						}
						else {
							if (res.size() != 0) res.erase(res.begin() + j);
						}
					}
				}
			}
			if (b) res.push_back(cur_obj);
		}
		return res;
	}
private:
	cv::dnn::Net net;
	int desired_class_id;
	float confidence_threshold;
	String net_input_name;
	String net_output_name;
	double net_scalefactor;
	Size net_size;
	Scalar net_mean;
	bool net_swapRB;
	vector<string> v;
	const string label[21] = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair" ,"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",  "train", "tvmonitor" };
};

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

int main(int argc, char** argv) {
	CommandLineParser parser(argc, argv, keys);
	cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();
	const string label[21] = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair" ,"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",  "train", "tvmonitor" };
	String video_name = parser.get<String>("video_name");
	int start_frame = parser.get<int>("start_frame");
	int frame_step = parser.get<int>("frame_step");
	/*String detector_model = parser.get<String>("detector_model");
	String detector_weights = parser.get<String>("detector_weights");
	String classification_model = parser.get<String>("classification_model");
	String classification_weights = parser.get<String>("classification_weights");*/
	int desired_class_id = parser.get<int>("desired_class_id");
	desired_class_id = -1;
	video_name = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/data/topdogs.mp4";
	detector_weights = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel";
	detector_model = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/mobilenet-ssd/caffe/mobilenet-ssd.prototxt";
	classification_weights = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/squeezenet/1.1/caffe/squeezenet1.1.caffemodel";
	classification_model = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/squeezenet/1.1/caffe/squeezenet1.1.prototxt";
	string labels = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/data/squeezenet1.1.labels";
	std::vector<string> v;
	v = { "cat", "dog" };
	v = {};
	if (video_name.empty() || detector_model.empty() || detector_weights.empty()) {
		help();
		return -1;
	}
	start_frame = 210;
	frame_step = 2;



	string st; // сюда будем класть считанные строки
	ifstream file(labels); // файл из которого читаем (для линукс путь будет выглядеть по другому)
	///
	while (getline(file, st)) { // пока не достигнут конец файла класть очередную строку в переменную (s)
		lab.push_back(st);
	}
	file.close();


	//open the capture
	VideoCapture cap;
	cap.open(video_name);
	cap.set(CAP_PROP_POS_FRAMES, start_frame);

	if (!cap.isOpened()) {
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		parser.printMessage();
		return -1;
	}

	// If you use the popular MobileNet_SSD detector, the default parameters may be used.
	// Otherwise, set your own parameters (net_mean, net_scalefactor, etc).
	DnnObjectDetector detector(detector_model, detector_weights, desired_class_id, v);

	Mat frame;
	namedWindow("Tracking by Matching", 1);

	int frame_counter = -1;
	int64 time_total = 0;
	bool paused = false;
	for (;;)
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

		TrackedObjects detections = detector.detect(frame, frame_counter);

		// timestamp in milliseconds
		uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / 30 * frame_counter);
		tracker->process(frame, detections, cur_timestamp);

		frame_time = getTickCount() - frame_time;
		time_total += frame_time;

		// Drawing colored "worms" (tracks).
		frame = tracker->drawActiveTracks(frame);


		// Drawing all detected objects on a frame by BLUE COLOR
		for (const auto &detection : detections) {
			if (detection.frame_idx == -1) {
				cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 1);

			}
		}



		// Drawing tracked detections only by RED color and print ID and detection
		// confidence level.
		for (const auto &detection : tracker->trackedDetections()) {

			if ((detection.class_id_21 == 12) && (detection.saved == false) && (detection.object_id != -1)) {
				string path = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/data/" + std::to_string(detection.object_id) + "_" + lab[detection.class_id] + ".jpg";
				Mat res_mat(frame, detection.rect);
				InputArray res(res_mat);
				imwrite(path, res);
				detection.saved = true;
			}
			
			cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 1);
			std::string text = std::to_string(detection.object_id) + "/" + detection.class_string +
				" conf: " + std::to_string(detection.confidence);
			cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_COMPLEX,
				0.5, cv::Scalar(0, 0, 255), 1);
			//cout << detection.object_id << " / " <<detection.class_id << " " << detection.class_string << endl;

		}

		imshow("Tracking by Matching", frame);

		char c = (char)waitKey(2);
		if (c == 'q')
			break;
		if (c == 'p')
			paused = !paused;

	}

	double s_i_dont_know = frame_counter / (time_total / getTickFrequency());
	printf("FPS: %f\n", s_i_dont_know);

	return 0;
}
