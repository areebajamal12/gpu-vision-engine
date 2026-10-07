#include "environment.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef GVE_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace gve::benchmark {
namespace {
std::string command_output(const char* command) {
  std::array<char, 256> buffer{};
  std::string result;
  if (FILE* pipe = popen(command, "r")) {
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) result += buffer.data();
    pclose(pipe);
  }
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
  return result;
}
}  // namespace

std::string cpu_model() {
#ifdef __APPLE__
  size_t size = 0;
  sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0);
  std::string value(size, '\0');
  if (size && sysctlbyname("machdep.cpu.brand_string", value.data(), &size, nullptr, 0) == 0) {
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
  }
  return command_output("sysctl -n hw.model 2>/dev/null");
#else
  std::ifstream file("/proc/cpuinfo");
  std::string line;
  while (std::getline(file, line)) {
    const auto marker = line.find("model name");
    const auto colon = line.find(':');
    if (marker != std::string::npos && colon != std::string::npos) return line.substr(colon + 2);
  }
  return "unavailable";
#endif
}

std::string gpu_model() {
#ifdef GVE_HAS_CUDA
  cudaDeviceProp properties{};
  return cudaGetDeviceProperties(&properties, 0) == cudaSuccess ? properties.name : "unavailable";
#else
  return "not enabled";
#endif
}

std::string cuda_runtime_version() {
#ifdef GVE_HAS_CUDA
  int version = 0;
  if (cudaRuntimeGetVersion(&version) != cudaSuccess) return "unavailable";
  return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
#else
  return "not enabled";
#endif
}

std::string cuda_toolkit_version() {
#ifdef GVE_HAS_CUDA
  return std::to_string(CUDART_VERSION / 1000) + "." + std::to_string((CUDART_VERSION % 1000) / 10);
#else
  return "not enabled";
#endif
}

std::string cuda_driver_version() {
#ifdef GVE_HAS_CUDA
  int version = 0;
  if (cudaDriverGetVersion(&version) != cudaSuccess) return "unavailable";
  return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
#else
  return "not enabled";
#endif
}

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (char c : value) {
    if (c == '"' || c == '\\')
      output << '\\' << c;
    else if (c == '\n')
      output << "\\n";
    else
      output << c;
  }
  return output.str();
}
}  // namespace gve::benchmark
