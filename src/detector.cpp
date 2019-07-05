#include "detector.h"

#include <fstream>
#include <string>


std::string readLabel(const String& path, int number) {
	std::ifstream is(path);
	std::string temp;
	std::vector<std::string> result;
	while (std::getline(is, temp)) {
		if (temp.size()) {
			result.push_back(temp);
		}
	}
	return number < result.size() ? result[number] : "Undef";
}


DnnDetector::DnnDetector(const String & ptm, const String & ptc, const String& ptl) {
	net = readNet(ptm, ptc);
	this->ptl = ptl;

	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}


vector<DetectedObject> DnnDetector::Detect(Mat image, Size size, Scalar mean, bool swapRB, double scale) {
	ResizeFilter resized(size.width, size.height);

	net.setInput(blobFromImage(resized.ProcessImage(image), scale, size, mean, swapRB));

	Mat reshaped = net.forward().reshape(1);

	vector<DetectedObject> result;
	for (int i = 0; i < reshaped.rows; ++i) {
		uchar* ptr = reshaped.data + i * reshaped.step;

		DetectedObject temp;		
		temp.classname = readLabel(ptl, ptr[1]);
		temp.uuid = ptr[1];
		temp.Left = ptr[3];
 		temp.Top = ptr[6];
		temp.Right = ptr[5];
		temp.Bottom = ptr[4];

		result.push_back(temp);
	}

	return result;
}
