#pragma once

#include <string>
#include <iostream>
#include <sstream>

namespace mps {
namespace util {

enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error
};

class Logger {
public:
    // Get singleton instance
    static Logger& GetInstance();

    // Set minimum log level
    void SetLogLevel(LogLevel level);

    // Log functions
    void Debug(const std::string& message);
    void Info(const std::string& message);
    void Warning(const std::string& message);
    void Error(const std::string& message);

    // Template version for formatted logging
    template<typename... Args>
    void Debug(Args&&... args) {
        if (min_level_ <= LogLevel::Debug) {
            Log(LogLevel::Debug, Format(std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void Info(Args&&... args) {
        if (min_level_ <= LogLevel::Info) {
            Log(LogLevel::Info, Format(std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void Warning(Args&&... args) {
        if (min_level_ <= LogLevel::Warning) {
            Log(LogLevel::Warning, Format(std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void Error(Args&&... args) {
        if (min_level_ <= LogLevel::Error) {
            Log(LogLevel::Error, Format(std::forward<Args>(args)...));
        }
    }

private:
    Logger() = default;
    ~Logger() = default;

    // Delete copy/move constructors
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    void Log(LogLevel level, const std::string& message);

    // Variadic template for formatting
    template<typename... Args>
    std::string Format(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }

    LogLevel min_level_ = LogLevel::Info;
};

// Convenience functions
inline void LogDebug(const std::string& msg) { Logger::GetInstance().Debug(msg); }
inline void LogInfo(const std::string& msg) { Logger::GetInstance().Info(msg); }
inline void LogWarning(const std::string& msg) { Logger::GetInstance().Warning(msg); }
inline void LogError(const std::string& msg) { Logger::GetInstance().Error(msg); }

template<typename... Args>
void LogDebug(Args&&... args) { Logger::GetInstance().Debug(std::forward<Args>(args)...); }

template<typename... Args>
void LogInfo(Args&&... args) { Logger::GetInstance().Info(std::forward<Args>(args)...); }

template<typename... Args>
void LogWarning(Args&&... args) { Logger::GetInstance().Warning(std::forward<Args>(args)...); }

template<typename... Args>
void LogError(Args&&... args) { Logger::GetInstance().Error(std::forward<Args>(args)...); }

}  // namespace util
}  // namespace mps
