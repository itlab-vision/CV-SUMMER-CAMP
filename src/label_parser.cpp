#include "label_parser.h"

int getClassNumber(std::string pathToLabels) {
	int classNumber;
	std::ifstream read(pathToLabels);
	std::string file;
	read >> file;
	std::cout << file << std::endl;







	return classNumber;
}