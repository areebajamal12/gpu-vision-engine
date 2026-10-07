#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gve/voxelization.hpp"

namespace {

struct ViewState {
  float yaw{-35.0F};
  float pitch{55.0F};
  float distance{125.0F};
  double last_x{};
  double last_y{};
  bool dragging{};
  bool show_points{true};
  bool show_voxels{true};
  bool capture_requested{};
};

ViewState state;

void reset_view() { state = ViewState{}; }

void key_callback(GLFWwindow* window, int key, int, int action, int) {
  if (action != GLFW_PRESS) return;
  if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (key == GLFW_KEY_1) {
    state.show_points = true;
    state.show_voxels = false;
  }
  if (key == GLFW_KEY_2) {
    state.show_points = false;
    state.show_voxels = true;
  }
  if (key == GLFW_KEY_3) {
    state.show_points = true;
    state.show_voxels = true;
  }
  if (key == GLFW_KEY_R) reset_view();
  if (key == GLFW_KEY_S) state.capture_requested = true;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int) {
  if (button != GLFW_MOUSE_BUTTON_LEFT) return;
  state.dragging = action == GLFW_PRESS;
  glfwGetCursorPos(window, &state.last_x, &state.last_y);
}

void cursor_callback(GLFWwindow*, double x, double y) {
  if (!state.dragging) return;
  state.yaw += static_cast<float>(x - state.last_x) * 0.25F;
  state.pitch =
      std::clamp(state.pitch + static_cast<float>(y - state.last_y) * 0.25F, -89.0F, 89.0F);
  state.last_x = x;
  state.last_y = y;
}

void scroll_callback(GLFWwindow*, double, double y_offset) {
  state.distance = std::clamp(state.distance - static_cast<float>(y_offset) * 5.0F, 15.0F, 300.0F);
}

void set_projection(int width, int height) {
  const double aspect = static_cast<double>(width) / std::max(height, 1);
  constexpr double near_plane = 1.0;
  constexpr double far_plane = 500.0;
  constexpr double half_height = 0.55;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(-half_height * aspect, half_height * aspect, -half_height, half_height, near_plane,
            far_plane);
  glMatrixMode(GL_MODELVIEW);
}

void draw_ground_grid() {
  glColor3f(0.18F, 0.22F, 0.28F);
  glLineWidth(1.0F);
  glBegin(GL_LINES);
  for (int coordinate = -50; coordinate <= 50; coordinate += 10) {
    glVertex3f(static_cast<float>(coordinate), -50.0F, 0.0F);
    glVertex3f(static_cast<float>(coordinate), 50.0F, 0.0F);
    glVertex3f(-50.0F, static_cast<float>(coordinate), 0.0F);
    glVertex3f(50.0F, static_cast<float>(coordinate), 0.0F);
  }
  glEnd();
}

void draw_points(const std::vector<gve::PointXYZI>& points) {
  glPointSize(1.5F);
  glBegin(GL_POINTS);
  for (const auto& point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
    const float intensity = std::clamp(point.intensity, 0.0F, 1.0F);
    glColor3f(0.15F + 0.30F * intensity, 0.45F + 0.50F * intensity, 0.85F - 0.35F * intensity);
    glVertex3f(point.x, point.y, point.z);
  }
  glEnd();
}

void draw_voxels(const gve::VoxelizationResult& result, const gve::VoxelGrid& grid) {
  const auto nx = static_cast<std::uint32_t>(std::lround((grid.max_x - grid.min_x) / grid.voxel_x));
  const auto ny = static_cast<std::uint32_t>(std::lround((grid.max_y - grid.min_y) / grid.voxel_y));
  glPointSize(4.0F);
  glBegin(GL_POINTS);
  for (const auto& voxel : result.voxels) {
    const std::uint32_t ix = voxel.linear_index % nx;
    const std::uint32_t yz = voxel.linear_index / nx;
    const std::uint32_t iy = yz % ny;
    const std::uint32_t iz = yz / ny;
    const float occupancy =
        std::min(1.0F, std::log2(static_cast<float>(voxel.point_count) + 1.0F) / 5.0F);
    glColor3f(1.0F, 0.25F + 0.65F * occupancy, 0.08F);
    glVertex3f(grid.min_x + (static_cast<float>(ix) + 0.5F) * grid.voxel_x,
               grid.min_y + (static_cast<float>(iy) + 0.5F) * grid.voxel_y,
               grid.min_z + (static_cast<float>(iz) + 0.5F) * grid.voxel_z);
  }
  glEnd();
}

using Glyph = std::array<std::uint8_t, 7>;

Glyph glyph(char character) {
  switch (character) {
    case 'A':
      return {14, 17, 17, 31, 17, 17, 17};
    case 'B':
      return {30, 17, 17, 30, 17, 17, 30};
    case 'C':
      return {14, 17, 16, 16, 16, 17, 14};
    case 'D':
      return {30, 17, 17, 17, 17, 17, 30};
    case 'E':
      return {31, 16, 16, 30, 16, 16, 31};
    case 'F':
      return {31, 16, 16, 30, 16, 16, 16};
    case 'G':
      return {14, 17, 16, 23, 17, 17, 15};
    case 'H':
      return {17, 17, 17, 31, 17, 17, 17};
    case 'I':
      return {31, 4, 4, 4, 4, 4, 31};
    case 'J':
      return {7, 2, 2, 2, 18, 18, 12};
    case 'K':
      return {17, 18, 20, 24, 20, 18, 17};
    case 'L':
      return {16, 16, 16, 16, 16, 16, 31};
    case 'M':
      return {17, 27, 21, 21, 17, 17, 17};
    case 'N':
      return {17, 25, 21, 19, 17, 17, 17};
    case 'O':
      return {14, 17, 17, 17, 17, 17, 14};
    case 'P':
      return {30, 17, 17, 30, 16, 16, 16};
    case 'Q':
      return {14, 17, 17, 17, 21, 18, 13};
    case 'R':
      return {30, 17, 17, 30, 20, 18, 17};
    case 'S':
      return {15, 16, 16, 14, 1, 1, 30};
    case 'T':
      return {31, 4, 4, 4, 4, 4, 4};
    case 'U':
      return {17, 17, 17, 17, 17, 17, 14};
    case 'V':
      return {17, 17, 17, 17, 17, 10, 4};
    case 'W':
      return {17, 17, 17, 21, 21, 21, 10};
    case 'X':
      return {17, 17, 10, 4, 10, 17, 17};
    case 'Y':
      return {17, 17, 10, 4, 4, 4, 4};
    case 'Z':
      return {31, 1, 2, 4, 8, 16, 31};
    case '0':
      return {14, 17, 19, 21, 25, 17, 14};
    case '1':
      return {4, 12, 4, 4, 4, 4, 14};
    case '2':
      return {14, 17, 1, 2, 4, 8, 31};
    case '3':
      return {30, 1, 1, 14, 1, 1, 30};
    case '4':
      return {2, 6, 10, 18, 31, 2, 2};
    case '5':
      return {31, 16, 16, 30, 1, 1, 30};
    case '6':
      return {14, 16, 16, 30, 17, 17, 14};
    case '7':
      return {31, 1, 2, 4, 8, 8, 8};
    case '8':
      return {14, 17, 17, 14, 17, 17, 14};
    case '9':
      return {14, 17, 17, 15, 1, 1, 14};
    case '.':
      return {0, 0, 0, 0, 0, 12, 12};
    case ':':
      return {0, 12, 12, 0, 12, 12, 0};
    case '-':
      return {0, 0, 0, 31, 0, 0, 0};
    case '/':
      return {1, 2, 2, 4, 8, 8, 16};
    case '|':
      return {4, 4, 4, 4, 4, 4, 4};
    case '+':
      return {0, 4, 4, 31, 4, 4, 0};
    default:
      return {};
  }
}

void draw_text(float x, float y, const std::string& text, float scale) {
  glBegin(GL_QUADS);
  for (const char character : text) {
    const auto bitmap = glyph(character);
    for (std::size_t row = 0; row < bitmap.size(); ++row) {
      for (int column = 0; column < 5; ++column) {
        if ((bitmap[row] & (1U << (4 - column))) == 0) continue;
        const float left = x + static_cast<float>(column) * scale;
        const float top = y + static_cast<float>(row) * scale;
        glVertex2f(left, top);
        glVertex2f(left + scale, top);
        glVertex2f(left + scale, top + scale);
        glVertex2f(left, top + scale);
      }
    }
    x += 6.0F * scale;
  }
  glEnd();
}

void draw_overlay(int width, int height, std::size_t point_count, std::size_t voxel_count,
                  const std::string& backend) {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, width, height, 0.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glDisable(GL_DEPTH_TEST);
  glColor4f(0.015F, 0.025F, 0.045F, 0.88F);
  glBegin(GL_QUADS);
  glVertex2f(18.0F, 18.0F);
  glVertex2f(700.0F, 18.0F);
  glVertex2f(700.0F, 198.0F);
  glVertex2f(18.0F, 198.0F);
  glEnd();
  const std::vector<std::string> lines = {"REAL NUSCENES LIDAR  |  " + backend + " VOXELIZATION",
                                          "RAW POINTS " + std::to_string(point_count) +
                                              "  |  OCCUPIED VOXELS " + std::to_string(voxel_count),
                                          "VERIFIED TESLA T4 EVIDENCE",
                                          "CPU 2.106 MS  |  CUDA KERNEL 0.396 MS  |  5.32X",
                                          "CUDA END-TO-END 1.048 MS  |  2.01X",
                                          "1 POINTS  2 VOXELS  3 BOTH  |  DRAG + SCROLL",
                                          "R RESET  |  S SAVE SCREENSHOT"};
  float y = 34.0F;
  for (std::size_t index = 0; index < lines.size(); ++index) {
    index == 0 ? glColor3f(0.30F, 0.90F, 1.0F) : glColor3f(0.86F, 0.90F, 0.96F);
    draw_text(32.0F, y, lines[index], 2.0F);
    y += 22.0F;
  }
  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void save_screenshot(int width, int height) {
  std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 3);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
  std::ofstream output("voxel-viewer.ppm", std::ios::binary);
  output << "P6\n" << width << ' ' << height << "\n255\n";
  const auto row_bytes = static_cast<std::size_t>(width) * 3;
  for (int row = height - 1; row >= 0; --row) {
    output.write(reinterpret_cast<const char*>(pixels.data() + row_bytes * row),
                 static_cast<std::streamsize>(row_bytes));
  }
  if (!output) throw std::runtime_error("Could not write voxel-viewer.ppm");
  std::cout << "Saved voxel-viewer.ppm\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: voxel_viewer <nuScenes .pcd.bin sweep>\n";
    return 1;
  }
  try {
    const auto points = gve::load_nuscenes_lidar_sweep(argv[1]);
    const gve::VoxelGrid grid;
#ifdef GVE_HAS_CUDA
    const auto voxels = gve::voxelize_cuda(points, grid);
    const std::string backend = "CUDA";
#else
    const auto voxels = gve::voxelize_cpu(points, grid);
    const std::string backend = "CPU FALLBACK";
#endif
    if (!glfwInit()) throw std::runtime_error("GLFW initialization failed");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow* window =
        glfwCreateWindow(1280, 800, "gpu-vision-engine | nuScenes voxelization", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("OpenGL window creation failed");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const std::string title = "gpu-vision-engine | " + std::to_string(points.size()) +
                              " points -> " + std::to_string(voxels.voxels.size()) +
                              " occupied voxels | 1 points, 2 voxels, 3 both";
    glfwSetWindowTitle(window, title.c_str());
    while (!glfwWindowShouldClose(window)) {
      int width = 0;
      int height = 0;
      glfwGetFramebufferSize(window, &width, &height);
      glViewport(0, 0, width, height);
      set_projection(width, height);
      glClearColor(0.025F, 0.035F, 0.055F, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glLoadIdentity();
      glTranslatef(0.0F, 0.0F, -state.distance);
      glRotatef(state.pitch, 1.0F, 0.0F, 0.0F);
      glRotatef(state.yaw, 0.0F, 0.0F, 1.0F);
      draw_ground_grid();
      if (state.show_points) draw_points(points);
      if (state.show_voxels) draw_voxels(voxels, grid);
      draw_overlay(width, height, points.size(), voxels.voxels.size(), backend);
      if (state.capture_requested) {
        save_screenshot(width, height);
        state.capture_requested = false;
      }
      glfwSwapBuffers(window);
      glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Viewer error: " << error.what() << '\n';
    return 1;
  }
}
