/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-31 21:43:53 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-07-31 21:43:53 
 */

#ifndef _LOGGING_H_
#define _LOGGING_H_
#pragma warning( disable : 4996 )


#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>

namespace ocr
{


enum class LogLevel {
    LOG_LEVEL_NONE,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR
};

class LogStreamBuffer : public std::stringbuf {
private:
    LogLevel log_level_;
public:
    LogStreamBuffer() :log_level_(LogLevel::LOG_LEVEL_INFO) {}
    LogStreamBuffer(LogLevel log_level) :log_level_(log_level) {}

    LogStreamBuffer(const LogStreamBuffer& other) = delete;
    LogStreamBuffer(LogStreamBuffer& other) = delete;
    LogStreamBuffer& operator=(const LogStreamBuffer& other) = delete;
    LogStreamBuffer& operator=(LogStreamBuffer&& other) = delete;

    ~LogStreamBuffer() override {
        if (pbase() != pptr()) {
            putOutput();
        }
    }

    int32_t sync() override {
        putOutput();
        return 0;
    }

    void putOutput() {
        std::time_t timestamp = std::time(nullptr);
        tm* tm_local = std::localtime(&timestamp);
        std::cout << "[";
        std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
        std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";

        switch(log_level_) {
        case LogLevel::LOG_LEVEL_NONE:
            break;
        case LogLevel::LOG_LEVEL_INFO:
            std::cout << "[" << "I" << "]" << " ";
            break;
        case LogLevel::LOG_LEVEL_WARNING:
            std::cout << "[" << "E" << "]" << " ";
            break;
        case LogLevel::LOG_LEVEL_ERROR:
            std::cout << "[" << "E" << "]" << " ";
            break;
        }
        std::cout << str();
        str("");
    }
};


class OnnxLog : public std::ostream {
private:
    LogStreamBuffer buffer_;
	std::string log_msg_;

public:
    OnnxLog() : buffer_(LogLevel::LOG_LEVEL_INFO), std::ostream(&buffer_), log_msg_{} {}
    OnnxLog(LogLevel log_level) : buffer_(log_level), std::ostream(&buffer_), log_msg_{} {}

    OnnxLog(const OnnxLog& other) = delete;
    OnnxLog(OnnxLog&& other) = delete;
    OnnxLog& operator=(const OnnxLog& other) = delete;
    OnnxLog& operator=(OnnxLog&& other) = delete;
};



}

#endif