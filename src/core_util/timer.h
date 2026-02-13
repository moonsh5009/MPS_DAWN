#pragma once

#include "core_util/types.h"
#include <chrono>
#include <string>

namespace mps {
namespace util {

class Timer {
public:
    Timer();

    // Start/restart the timer
    void Start();

    // Stop the timer
    void Stop();

    // Reset the timer
    void Reset();

    // Get elapsed time in seconds
    float64 GetElapsedSeconds() const;

    // Get elapsed time in milliseconds
    float64 GetElapsedMilliseconds() const;

    // Get elapsed time in microseconds
    float64 GetElapsedMicroseconds() const;

    // Check if timer is running
    bool IsRunning() const { return is_running_; }

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint start_time_;
    TimePoint stop_time_;
    bool is_running_;
};

// Scoped timer for automatic profiling
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

    // Delete copy/move
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

private:
    std::string name_;
    Timer timer_;
};

}  // namespace util
}  // namespace mps
