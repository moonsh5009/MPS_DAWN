#include "core_util/timer.h"
#include "core_util/logger.h"

namespace mps {
namespace util {

Timer::Timer()
    : start_time_(Clock::now())
    , stop_time_(start_time_)
    , is_running_(false) {
}

void Timer::Start() {
    start_time_ = Clock::now();
    is_running_ = true;
}

void Timer::Stop() {
    stop_time_ = Clock::now();
    is_running_ = false;
}

void Timer::Reset() {
    start_time_ = Clock::now();
    stop_time_ = start_time_;
    is_running_ = false;
}

double Timer::GetElapsedSeconds() const {
    TimePoint end_time = is_running_ ? Clock::now() : stop_time_;
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
        end_time - start_time_);
    return duration.count();
}

double Timer::GetElapsedMilliseconds() const {
    TimePoint end_time = is_running_ ? Clock::now() : stop_time_;
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        end_time - start_time_);
    return duration.count();
}

double Timer::GetElapsedMicroseconds() const {
    TimePoint end_time = is_running_ ? Clock::now() : stop_time_;
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
        end_time - start_time_);
    return duration.count();
}

// ScopedTimer implementation
ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name) {
    timer_.Start();
    LogDebug("[Profile] ", name_, " started");
}

ScopedTimer::~ScopedTimer() {
    timer_.Stop();
    LogDebug("[Profile] ", name_, " finished in ",
             timer_.GetElapsedMilliseconds(), " ms");
}

}  // namespace util
}  // namespace mps
