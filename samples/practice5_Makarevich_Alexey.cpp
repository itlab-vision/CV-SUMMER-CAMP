#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "tracking_by_matching.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <classificator.h>
#include <map>
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::tbm;

const char* cmdAbout =
"This is an empty application that can be treated as a template for your "
"own doing-something-cool applications.";

const char* cmdOptions =
"{video_name      | | video name               }"
"{ h ? help usage | | print help message       }";

String mobilenet_model = "C:\\CV-SUMMER-CAMP\\data\\mobilenet-ssd.caffemodel";
String mobilenet_weights = "C:\\CV-SUMMER-CAMP\\data\\mobilenet-ssd.prototxt";
String mobile_labels = "C:\\CV-SUMMER-CAMP\data\\labels.txt";
String squeezenet_model = "C:\\CV-SUMMER-CAMP\\data\\squeezenet1.1.caffemodel";
String squeezene_weights = "C:\\CV-SUMMER-CAMP\\data\\squeezenet1.1.prototxt";
String squeezene_labels = "C:\\CV-SUMMER-CAMP\\data\\squeezenetlabels.txt";
string result = "C:\CV-SUMMER-CAMP\data\result";
vector<int>desired_class_id = { 12 };//dogs

struct resultImage
{
	Mat img;
	float confidence;
	int count;
	resultImage() :img(), confidence(0), count(0) {}
	resultImage(Mat i, float c, int count) : img(i), confidence(c), count(count) {}

};

static void help()
{
	cout << "Oops.Somethings went wrong :C" << endl;
}
ifstream& GotoLine(ifstream& file, int num)
{
	file.seekg(ios::beg);
	for (int i = 0; i < num - 1; ++i) {
		file.ignore(numeric_limits<streamsize>::max(), '\n');
	}
	return file;
}
string getLabel(string label_path, int num)
{
	ifstream labels(label_path);
	string class_name;
	GotoLine(labels, num);
	getline(labels, class_name);
	return class_name;
}

cv::Ptr<ITrackerByMatching> createTrackerByMatchingWithFastDescriptor();

class DnnObjectDetector
{
public:
	DnnObjectDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
		std::vector<int> desired_class_id,
		float confidence_threshold = 0.2,
		//the following parameters are default for popular MobileNet_SSD caffe model
		const String& net_input_name = "data",
		const String& net_output_name = "detection_out",
		double net_scalefactor = 0.007843,
		const Size& net_size = Size(300, 300),
		const Scalar& net_mean = Scalar(127.5, 127.5, 127.5),
		bool net_swapRB = false)
		:desired_class_id(desired_class_id),
		confidence_threshold(confidence_threshold),
		net_input_name(net_input_name),
		net_output_name(net_output_name),
		net_scalefactor(net_scalefactor),
		net_size(net_size),
		net_mean(net_mean),
		net_swapRB(net_swapRB)
	{
		net = dnn::readNet(net_caffe_model_path, net_caffe_weights_path);
		if (net.empty())
			CV_Error(Error::StsError, "Cannot read Caffe net");
	}
	TrackedObjects detect(const cv::Mat& frame, int frame_idx, string label_path, std::vector<int> ids_for_tracking)
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
			float cur_confidence = detection_as_mat.at<float>(i, 2);
			int cur_class_id = static_cast<int>(detection_as_mat.at<float>(i, 1));
			int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * frame.cols);
			int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * frame.rows);
			int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * frame.cols);
			int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * frame.rows);
			string class_name = getLabel(label_path, cur_class_id);

			if (find(ids_for_tracking.begin(), ids_for_tracking.end(), cur_class_id) == ids_for_tracking.end())
			{
				continue;
			}
			Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));

			if (cur_confidence < confidence_threshold)
				continue;


			//clipping by frame size
			cur_rect = cur_rect & Rect(Point(), frame.size());
			if (cur_rect.empty())
				continue;

			TrackedObject cur_obj(cur_rect, cur_confidence, frame_idx, -1, class_name);
			res.push_back(cur_obj);
		}
		return res;
	}
private:
	cv::dnn::Net net;
	std::vector<int> desired_class_id;
	float confidence_threshold;
	String net_input_name;
	String net_output_name;
	double net_scalefactor;
	Size net_size;
	Scalar net_mean;
	bool net_swapRB;
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

int main(int argc, const char** argv) {
	cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();
	Classificator* dog_breed_classificator = new DnnClassificator(squeezenet_model, squeezene_weights, squeezene_labels, 227, 227, Scalar(0, 0, 0), true);
	// Parse command line arguments.
	CommandLineParser parser(argc, argv, cmdOptions);
	parser.about(cmdAbout);

	// If help option is given, print help message and exit.
	if (parser.get<bool>("help")) {
		parser.printMessage();
		return 0;
	}

	//something is not cool but working

	map<int, resultImage>dogsCounter;

	vector<int> checked_dogs;
	String video_name = parser.get<String>("video_name");
	//open the capture

	VideoCapture cap;
	cap.open(video_name);
	cap.set(CAP_PROP_POS_FRAMES, 100);
	if (!cap.isOpened())
	{
		help();
		return -1;
	}


	DnnObjectDetector detector(mobilenet_model, mobilenet_weights, desired_class_id);

	Mat frame;

	int frame_counter = -1;
	int64 time_total = 0;
	bool paused = false;
	for (;; )
	{
		cap >> frame;

		frame_counter++;
		if (frame.empty())
		{
			break;
		}
		if (frame_counter % 100 != 0)
			continue;
		

		frame_counter++;
		int64 frame_time = getTickCount();
		TrackedObjects detections = detector.detect(frame, frame_counter, mobile_labels, desired_class_id);
		uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / 30 * frame_counter);
		tracker->process(frame, detections, cur_timestamp);
		frame_time = getTickCount() - frame_time;
		time_total += frame_time;
		for (const auto &detection : detections)
		{
			if (find(checked_dogs.begin(), checked_dogs.end(), detection.object_id) != checked_dogs.end())
			{
				continue;
			}
				Point classIdPoint;
				double confidence;
				int classId;
				checked_dogs.push_back(detection.object_id);
				Mat res = dog_breed_classificator->Classify(frame(detection.rect));
				minMaxLoc(res, 0, &confidence, 0, &classIdPoint);
				classId = classIdPoint.x;
				if (dogsCounter.count(detection.class_id))
				{
					if (confidence > 0.3)
					{
						dogsCounter[classId].count++;
						if (dogsCounter[classId].confidence < confidence)
						{
							dogsCounter[classId].img = frame(detection.rect).clone();
							dogsCounter[classId].confidence = confidence;
							imshow(" ", frame(detection.rect).clone());
							waitKey();
						}
					}
				}
				else
				{
					if (confidence > 0.3)
					{
						dogsCounter[classId].count = 1;
						dogsCounter[classId].img = frame(detection.rect).clone();
						dogsCounter[classId].confidence = confidence;
						imshow(" ", frame(detection.rect).clone());
						waitKey();
					}
				}
				
			
			
		}

	}
	//output results
	ofstream fout;
	fout.open("C:\\CV-SUMMER-CAMP\\data\\result\\result.txt");
	for (auto x : dogsCounter)
	{
		fout << getLabel(squeezene_labels, x.first) + ":" + to_string(x.second.count);
		imwrite("C:\\CV-SUMMER-CAMP\\data\\result\\" + getLabel(squeezene_labels, x.first) + ".img", x.second.img);
	}


	return 0;
}
