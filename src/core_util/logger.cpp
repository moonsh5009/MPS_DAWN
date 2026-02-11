#include "core_util/logger.h"
#include <iostream>
#include <chrono>
#include <iomanip>

namespace mps {
namespace util {

Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::SetLogLevel(LogLevel level) {
    min_level_ = level;
}

void Logger::Debug(const std::string& message) {
    if (min_level_ <= LogLevel::Debug) {
        Log(LogLevel::Debug, message);
    }
}

void Logger::Info(const std::string& message) {
    if (min_level_ <= LogLevel::Info) {
        Log(LogLevel::Info, message);
    }
}

void Logger::Warning(const std::string& message) {
    if (min_level_ <= LogLevel::Warning) {
        Log(LogLevel::Warning, message);
    }
}

void Logger::Error(const std::string& message) {
    if (min_level_ <= LogLevel::Error) {
        Log(LogLevel::Error, message);
    }
}

void Logger::Log(LogLevel level, const std::string& message) {
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    // Format time
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    // Determine log level prefix and output stream
    const char* level_str;
    std::ostream* output = &std::cout;

    switch (level) {
        case LogLevel::Debug:
            level_str = "[DEBUG]";
            break;
        case LogLevel::Info:
            level_str = "[INFO]";
            break;
        case LogLevel::Warning:
            level_str = "[WARNING]";
            output = &std::cerr;
            break;
        case LogLevel::Error:
            level_str = "[ERROR]";
            output = &std::cerr;
            break;
        default:
            level_str = "[UNKNOWN]";
            break;
    }

    // Output log message
    *output << std::put_time(&tm_now, "%H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count()
            << " " << level_str << " " << message << std::endl;
}

}  // namespace util
}  // namespace mps
