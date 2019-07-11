#include "label_parser.h"

std::map<int, std::string> initializeClasses(std::string pathToLabels) {
    std::ifstream read(pathToLabels);
	std::map<int, std::string> dict;
	int i = 1;
	while (read >> dict[i]) {
		i++;
	}

	return dict;
}