#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "filter.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  src           | <none> | camera id or file path  }"
"{ q ? help usage   | <none> | print help message      }";

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

int main(int argc, char** argv)
{
	// Process input arguments
	CommandLineParser parser(argc, argv, cmdOptions);
	parser.about(cmdAbout);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	// Required parameter check
	if (!parser.has("src"))
	{
		cout << "\"src\" parameter is requred\n";
		return -1;
	}

    String src(parser.get<String>("src"));
	VideoCapture stream;

	// Stream or camera determination
    if (is_number(src))
		stream = VideoCapture(stoi(src));
	else
		stream = VideoCapture(src);	

	if (!stream.isOpened())
	{
		cout << "Unable to open stream/camera\n";
		return -1;
	}

	// Main loop
    while (true)
    {
        Mat frame;
		stream >> frame;
        if (frame.data == NULL) {
            break;
        }
        imshow("Output", frame);
        if (waitKey(1) >= 0) break;
    }
    waitKey(0);
    return 0;
}
