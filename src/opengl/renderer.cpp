
#include "empyrean/opengl/renderer.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "empyrean/utils/fps_counter.hpp"
#include "empyrean/utils/structs.hpp"

float scaleFactor = 1.0f;

float timeScale = 1.0f;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_Z && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    scaleFactor *= 1.1;
  }
  if (key == GLFW_KEY_X && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    scaleFactor *= 0.9;
  }
  if (key == GLFW_KEY_RIGHT_BRACKET && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    timeScale *= 2;
  }
  if (key == GLFW_KEY_LEFT_BRACKET && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    timeScale *= 0.5;
  }
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

void GlRenderer::processInput() {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
}

GlRenderer::GlRenderer(std::string title, int width, int height, int numBodies, int useGpu,
                       SharedData& sharedData)
    : width(width), height(height), numBodies(numBodies), sharedData(sharedData), useGpu(useGpu) {
  // glfw: initialize and configure
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // glfw: window creation
  window = glfwCreateWindow(width, height, title.c_str(), glfwGetPrimaryMonitor(), NULL);
  if (window == NULL) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwSetKeyCallback(window, key_callback);
  glfwSwapInterval(0);

  // glad: load all OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
  }
}

void GlRenderer::compileShaders() {
  const char* vertexShaderSource =
#include "vertex.shader"
      ;
  const char* fragmentShaderSource =
#include "fragment.shader"
      ;
  // build and compile shaders
  // vertex shader
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  // check for shader compile errors
  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << vertexShaderSource << std::endl;
  }
  // fragment shader
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  // check for shader compile errors
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
  }
  // link shaders
  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  // check for linking errors
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
  }
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
}

GlRenderer::~GlRenderer() {
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteProgram(shaderProgram);
  glfwTerminate();
}

void GlRenderer::initVertexData() {
  float vertices[3 * numBodies];
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void GlRenderer::mainLoop() {
  double previousTime = glfwGetTime();
  int frameCount = 0;

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  cudaGraphicsResource_t cudaResource;
  cudaError_t cudaStatus
      = cudaGraphicsGLRegisterBuffer(&cudaResource, VBO, cudaGraphicsMapFlagsNone);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaGraphicsGLRegisterBuffer error: " << cudaGetErrorString(cudaStatus)
              << std::endl;
  }
  // Map the CUDA graphics resource
  cudaGraphicsMapResources(1, &cudaResource);
  size_t size = numBodies * 3;
  // float* hostPointer = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));

  {
    std::lock_guard<std::mutex> lock(sharedData.get().mtx);
    cudaGraphicsResourceGetMappedPointer(&sharedData.get().devicePointer, &size, cudaResource);

    // sharedData.get().hostPointer = hostPointer;
    sharedData.get().cv.notify_one();
  }
  FpsCounter fpsCounter("OpenGL Renderer");
  while (!glfwWindowShouldClose(window)) {
    fpsCounter.displayFps();
    // input
    processInput();
    // render
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);
    glUniform1f(glGetUniformLocation(shaderProgram, "scaleFactor"), scaleFactor);
    glDrawArrays(GL_POINTS, 0, numBodies);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  // glUnmapBuffer(GL_ARRAY_BUFFER);
  sharedData.get().stopRequested = true;
  cudaGraphicsUnmapResources(1, &cudaResource);
  cudaGraphicsUnregisterResource(cudaResource);
}

void GlRenderer::setGlFlags() { glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); }

void GlRenderer::start() {
  setGlFlags();
  compileShaders();
  initVertexData();
  mainLoop();
}
