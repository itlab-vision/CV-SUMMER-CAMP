#pragma once
#include <string>

struct DetectedObject
{
    std::int32_t Left;
	std::int32_t Right;
	std::int32_t Top;
	std::int32_t Bottom;

	std::int32_t classId;
    std::string classname;

	std::float_t confidence;
};