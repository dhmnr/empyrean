
#include "empyrean/opengl/renderer.hpp"

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

void GlRenderer::processInput() {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
}

GlRenderer::GlRenderer(NbodyEngine* engine, const int width, const int height)
    : engine(engine), width(width), height(height) {
  objectcount = engine->numBodies;
  // glfw: initialize and configure
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // glfw: window creation
  window = glfwCreateWindow(width, height, "EMPYREAN", NULL, NULL);
  if (window == NULL) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwSwapInterval(0);

  // glad: load all OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
  }
}

void GlRenderer::compileShaders() {
  const char* vertexShaderSource
      = "#version 330 core\n"
        "layout (location = 0) in vec3 inPos;\n"
        "void main()\n"
        "{\n"
        "   gl_Position =  vec4(inPos, 1.0);\n"
        "   gl_PointSize = 2;\n"
        "}\0";
  const char* fragmentShaderSource
      = "#version 330 core\n"
        "out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
        "   FragColor = vec4(0.9f, 0.9f, 0.0f, 1.0f);\n"
        "}\n\0";
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
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
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
  float vertices[3 * objectcount];
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
  float* vertexData = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));

  while (!glfwWindowShouldClose(window)) {
    double currentTime = glfwGetTime();
    frameCount++;
    if (currentTime - previousTime >= 1.0) {
      std::cout << "FPS : " << frameCount << std::endl;

      frameCount = 0;
      previousTime = currentTime;
    }
    // input
    processInput();
    // render
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);

    engine->updatePositions(vertexData);
    glDrawArrays(GL_POINTS, 0, objectcount);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glUnmapBuffer(GL_ARRAY_BUFFER);
}

void GlRenderer::setGlFlags() { glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); }

void GlRenderer::start() {
  setGlFlags();
  compileShaders();
  initVertexData();
  mainLoop();
}
