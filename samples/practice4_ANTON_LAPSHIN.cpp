#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "tracking_by_matching.hpp"
#include <iostream>


using namespace std;
using namespace cv;
using namespace cv::tbm;

static const char* keys =
{ "{video_name       | | video name                       }"
"{start_frame      |0| Start frame                      }"
"{frame_step       |1| Frame step                       }"
"{detector_model   | | Path to detector's Caffe model   }"
"{detector_weights | | Path to detector's Caffe weights }"
"{desired_class_ids | | The desired classes that should be tracked }"
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
        "If `--desired_class_ids` parameter is set (id values separated by space),"
        " the detection result is filtered by class ids, returned by the detection network.\n"
        "(That is, if a detection net was trained on VOC dataset, then to track pedestrians point --desired_class_ids=15)\n"
        "Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
        "Call:\n"
        "./example_tracking_tracking_by_matching --video_name=<video_name> --detector_model=<detector_model_path> --detector_weights=<detector_weights_path> \\\n"
        "                                       [--start_frame=<start_frame>] \\\n"
        "                                       [--frame_step=<frame_step>] \\\n"
        "                                       [--desired_class_ids=<desired_class_ids>]\n"
        << endl;

    cout << "\n\nHot keys: \n"
        "\tq - quit the program\n"
        "\tp - pause/resume video\n";
}

cv::Ptr<ITrackerByMatching> createTrackerByMatchingWithFastDescriptor();

class DnnObjectDetector
{
public:
    DnnObjectDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
        String desired_class_ids_string = "",
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
        istringstream stream(desired_class_ids_string);
        while (1) {
            int n;
            stream >> n;
            if (!stream)
                break;
            desired_class_ids.push_back(n);
        }

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
            if ((desired_class_ids.size() > 0) &&
                find(desired_class_ids.begin(), desired_class_ids.end(), cur_class_id) == desired_class_ids.end())
                continue;

            //clipping by frame size
            cur_rect = cur_rect & Rect(Point(), frame.size());
            if (cur_rect.empty())
                continue;

            TrackedObject cur_obj(cur_rect, cur_confidence, frame_idx, -1);
            res.push_back(cur_obj);
        }
        return res;
    }
private:
    cv::dnn::Net net;
    vector<int> desired_class_ids;
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

int main(int argc, char** argv) {
    CommandLineParser parser(argc, argv, keys);
    cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();

    String video_name = parser.get<String>("video_name");
    int start_frame = parser.get<int>("start_frame");
    int frame_step = parser.get<int>("frame_step");
    String detector_model = parser.get<String>("detector_model");
    String detector_weights = parser.get<String>("detector_weights");
    String desired_class_ids = parser.get<String>("desired_class_ids");

    if (video_name.empty() || detector_model.empty() || detector_weights.empty())
    {
        help();
        return -1;
    }

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

    // If you use the popular MobileNet_SSD detector, the default parameters may be used.
    // Otherwise, set your own parameters (net_mean, net_scalefactor, etc).
    DnnObjectDetector detector(detector_model, detector_weights, desired_class_ids);

    Mat frame;
    namedWindow("Tracking by Matching", 1);

    int frame_counter = -1;
    int64 time_total = 0;
    bool paused = false;
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

        // timestamp in milliseconds
        uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / 30 * frame_counter);
        tracker->process(frame, detections, cur_timestamp);

        frame_time = getTickCount() - frame_time;
        time_total += frame_time;

        // Drawing colored "worms" (tracks).
        frame = tracker->drawActiveTracks(frame);


        // Drawing all detected objects on a frame by BLUE COLOR
        for (const auto &detection : detections) {
            cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
        }

        // Drawing tracked detections only and printing ID and detection confidence level.
        for (const auto &detection : tracker->trackedDetections()) {
            ostringstream confidencePercents;
            confidencePercents.precision(2);
            confidencePercents << fixed << detection.confidence * 100;

            srand(detection.object_id);
            int r = (rand() % 150) + 50;
            int g = (rand() % 150) + 50;
            int b = (rand() % 150) + 50;
            Scalar color = Scalar(r, g, b);

            cv::rectangle(frame, detection.rect, color, 3);

            string text = "id " + to_string(detection.object_id) + " (" + confidencePercents.str() + "%)";
            putText(frame, text, Point(detection.rect.x, detection.rect.y - 10), FONT_HERSHEY_DUPLEX, 1.2, Scalar(255, 255, 255), 3);
            putText(frame, text, Point(detection.rect.x, detection.rect.y - 10), FONT_HERSHEY_DUPLEX, 1.2, color, 1);
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
