
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <vector>

#include "empyrean/engine/body.hpp"
#include "empyrean/structs/initial_state.hpp"

InitialState parseInputYaml(std::string filename) {
  InitialState state;
  try {
    // Load the YAML file
    YAML::Node config = YAML::LoadFile(filename);

    // Parse units
    // YAML::Node units = config["units"];
    // std::string timeUnit = units["time"].as<std::string>();
    // std::string massUnit = units["mass"].as<std::string>();
    // std::string distanceUnit = units["distance"].as<std::string>();
    // std::string velocityUnit = units["velocity"].as<std::string>();

    // Parse global settings
    YAML::Node global = config["global"];
    double scaleFactor = std::stod(global["scale-factor"].as<std::string>());
    double timeStep = global["time-step"].as<double>();

    state.timeStep = timeStep;
    // Parse objects
    YAML::Node objects = config["objects"];
    YAML::Node staticObjects = objects["static"];
    for (const auto& object : staticObjects) {
      std::string name = object.begin()->first.as<std::string>();
      double mass = object[name]["mass"].as<double>();
      YAML::Node position = object[name]["position"];
      YAML::Node velocity = object[name]["velocity"];
      double posX = position[0].as<double>();
      double posY = position[1].as<double>();
      double posZ = position[2].as<double>();
      double velX = velocity[0].as<double>();
      double velY = velocity[1].as<double>();
      double velZ = velocity[2].as<double>();

      Body body({posX, posY, posZ}, mass, {velX, velY, velZ});

      state.bodies.push_back(body);
      //   std::cout << "Object: " << name << std::endl;
      //   std::cout << "  Mass: " << mass << std::endl;
      //   std::cout << "  Position: (" << posX << ", " << posY << ", " << posZ << ")" << std::endl;
      //   std::cout << "  Velocity: (" << velX << ", " << velY << ", " << velZ << ")" << std::endl;
    }
    state.objCount = state.bodies.size();
    state.scaleFactor = scaleFactor;

  } catch (const YAML::Exception& e) {
    std::cerr << "Error parsing YAML: " << e.what() << std::endl;
  }
  return state;
}
