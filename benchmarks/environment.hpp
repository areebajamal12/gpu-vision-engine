#pragma once
#include <string>
namespace gve::benchmark {
std::string cpu_model();
std::string gpu_model();
std::string cuda_runtime_version();
std::string cuda_toolkit_version();
std::string cuda_driver_version();
std::string json_escape(const std::string& value);
}  // namespace gve::benchmark
