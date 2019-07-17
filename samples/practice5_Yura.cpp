#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "tracking_by_matching.hpp"
#include "detector.h"
#include "classificator.h" 

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::tbm;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
{ "{video_name       | | video name                       }"
"{start_frame      |0| Start frame                      }"
"{frame_step       |1| Frame step                       }"
"{detector_model   | | Path to detector's Caffe model   }"
"{detector_weights | | Path to detector's Caffe weights }"
"{desired_class_id |-1| The desired class that should be tracked }"
"{desired_class_id |-1| The desired class that should be tracked }"
"{classificator_model   | | Path to classificator's Caffe model   }"
"{classificator_weights | | Path to classificator's Caffe weights }"
};
/*
-start_frame=200
-frame_step=4
-classificator_model="..\..\CV-SUMMER-CAMP\data\net\classification\squeezenet\1.1\caffe\squeezenet1.1.caffemodel"
-classificator_weights="..\..\CV-SUMMER-CAMP\data\net\classification\squeezenet\1.1\caffe\squeezenet1.1.prototxt"
-detector_model="C:\Users\temp2019\Desktop\CV-SUMMER-CAMP\data\object_detection\common\mobilenet-ssd\caffe\mobilenet-ssd.prototxt"
-detector_weights="C:\Users\temp2019\Desktop\CV-SUMMER-CAMP\data\object_detection\common\mobilenet-ssd\caffe\mobilenet-ssd.caffemodel"
-video_name="..\..\CV-SUMMER-CAMP\data\topdogs.mp4"
-desired_class_id=[12]
*/

//replace and delete later
class Dog
{
public:
	int breed;
	double confidence;
	Mat image;
};

class DnnObjectDetector
{
public:
	DnnObjectDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
		vector<int> &desired_class_id_vector,
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
		if (desired_class_id_vector.empty())
			desired_class_id.push_back(-1);
		else
			desired_class_id = desired_class_id_vector;
		net = dnn::readNetFromCaffe(net_caffe_model_path, net_caffe_weights_path);
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
			float cur_confidence = detection_as_mat.at<float>(i, 2);
			int cur_class_id = static_cast<int>(detection_as_mat.at<float>(i, 1));
			int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * frame.cols);
			int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * frame.rows);
			int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * frame.cols);
			int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * frame.rows);

			Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));

			if (cur_confidence < confidence_threshold)
				continue;
			if (desired_class_id[0] >= 0)
			{
				bool coincidence = false;
				for (int j : desired_class_id)
				{
					if (cur_class_id == j) {
						coincidence = true;
						break;
					}
				}
				if (!coincidence)
				{
					continue;
				}
			}

			//clipping by frame size
			cur_rect = cur_rect & Rect(Point(), frame.size());
			if (cur_rect.empty())
				continue;
			TrackedObject cur_obj(cur_rect, cur_confidence, frame_idx, -1, cur_class_id);
			res.push_back(cur_obj);
		}
		return res;
	}
private:
	cv::dnn::Net net;
	vector<int> desired_class_id;
	float confidence_threshold;
	String net_input_name;
	String net_output_name;
	double net_scalefactor;
	Size net_size;
	Scalar net_mean;
	bool net_swapRB;
};

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

int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);

  cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();

  String video_name = parser.get<String>("video_name");
  int start_frame = parser.get<int>("start_frame");
  int frame_step = parser.get<int>("frame_step");
  String detector_model = parser.get<String>("detector_model");
  String detector_weights = parser.get<String>("detector_weights");

  String classificator_model = parser.get<string>("classificator_model");
  String classificator_weights = parser.get<String>("classificator_weights");
  string class_ids = parser.get<string>("desired_class_id");
  vector<int> desired_class_id;
  int t = 0;
  for (int i = 1; i < class_ids.length(); i++)
  {
	  int t = 1;
	  if (class_ids[i] == ' ' || class_ids[i] == ']')
	  {
		  desired_class_id.push_back(stoi(string(class_ids, t, i - t)));
		  t = i;
	  }

  }

  if (video_name.empty() || detector_model.empty() || detector_weights.empty())
  {
	  return -1;
  }

  //open the capture
  VideoCapture cap;
  cap.open(video_name);
  cap.set(CAP_PROP_POS_FRAMES, start_frame);

  if (!cap.isOpened())
  {
	  cout << "***Could not initialize capturing...***\n";
	  cout << "Current parameter's value: \n";
	  parser.printMessage();
	  return -1;
  }

  // Do something cool.
  DnnObjectDetector detector(detector_model, detector_weights, desired_class_id);
  DnnClassificator dnn(classificator_model,classificator_weights,
	  300, 300, Scalar(127.5,127.5,127.5), true);

  Mat frame;
  namedWindow("Tracking by Matching", 1);

  int frame_counter = -1;
  int64 time_total = 0;
  bool paused = false;
  vector<Dog> dogs;
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

	  TrackedObjects detections = detector.detect(frame, frame_counter);

	  //Image classification	
	  for (auto i : detections)
	  {
		  if (i.class_id == 8 && i.confidence >=0.5)
		  {
			  bool was = false;
			  Dog temp;
			  Mat dnnimage = dnn.Classify(Mat(frame, i.rect));
			  //Show result
			  Point classIdPoint;
			  minMaxLoc(temp.image, 0, &temp.confidence, 0, &classIdPoint);
			  temp.breed = classIdPoint.x;
			  for (auto j : dogs)
			  {
				  if (temp.breed == j.breed)
				  {
					  was = true;
					  if (temp.confidence > j.confidence)
					  {
						  j.image = temp.image;
					  }
				  }
			  }
			  if (!was)
				dogs.push_back(temp);
		  }
	  }
	  
	  

	  // timestamp in milliseconds
	  uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / 30 * frame_counter);
	  tracker->process(frame, detections, cur_timestamp);

	  frame_time = getTickCount() - frame_time;
	  time_total += frame_time;

	  // Drawing colored "worms" (tracks).
	  frame = tracker->drawActiveTracks(frame);


	  // Drawing all detected objects on a frame by BLUE COLOR
	  for (const auto &detection : detections) {
		  cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 1);
	  }

	  // Drawing tracked detections only by RED color and print ID and detection
	  // confidence level.
	  for (const auto &detection : tracker->trackedDetections()) {
		  cv::rectangle(frame, detection.rect, cv::Scalar(0, 255, 0), 1);
		  std::string text = std::to_string(detection.class_id) + " id:" + std::to_string(detection.object_id) +
			  " conf: " + std::to_string(detection.confidence);
		  cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_COMPLEX_SMALL,
			  1.0, cv::Scalar(0, 0, 255), 1);
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
  
  for (auto i : dogs)
  {
	  imwrite(std::to_string(i.breed) + " " + std::to_string(i.confidence), i.image);
  }
  return 0;
}