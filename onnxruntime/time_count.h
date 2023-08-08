/*
 * @Author: OCR_J 
 * @Date: 2023-05-05 15:24:01 
 * @Last Modified by:   OCR_J 
 * @Last Modified time: 2023-05-05 15:24:01 
 * 
 * https://github.com/JHC521PJJ/tensorrt
 */

#ifndef __TIME_H__
#define __TIME_H__

#include <chrono>
#include <iostream>

class TimeCount {
public:
	using SystemClock = std::chrono::system_clock;
	using TimePoint = std::chrono::system_clock::time_point;
private:
	TimePoint time_;
private:
	TimeCount() :time_(TimePoint::min()) {}
public:
	TimeCount(const TimeCount& other) = delete;
	TimeCount(TimeCount&& other) = delete;
	TimeCount& operator=(const TimeCount& other) = delete;
	TimeCount& operator=(TimeCount&& other) = delete;

	static TimeCount& instance() {
		static TimeCount time_count;
		return time_count;
	}

	void start() noexcept { time_ = SystemClock::now(); }
	void clear() noexcept { time_ = TimePoint::min(); }
	bool isStarted() const noexcept {
		return (time_.time_since_epoch() != SystemClock::duration(0));
	}

	double getTime() const noexcept {
		if (isStarted()) {
			SystemClock::duration diff;
			diff = SystemClock::now() - time_;
			return std::chrono::duration<double, std::milli>(diff).count();
		}
		return static_cast<double>(0.0);
	}

	void printTime() const noexcept {
		std::cout << "Take time: " << getTime() << " ms" << "\n";
	}
};
#endif
