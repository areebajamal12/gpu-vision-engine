#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: voxel_viewer <nuScenes .pcd.bin sweep>\n";
    return 1;
  }
  try {
    const auto points = gve::load_nuscenes_lidar_sweep(argv[1]);
    const gve::VoxelGrid grid;
    const auto voxels = gve::voxelize_cpu(points, grid);
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
