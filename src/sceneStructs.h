#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
  SPHERE,
  CUBE,
  MESH,
};

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

struct Geom {
  enum GeomType type;
  int materialid;
  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;
  glm::mat4 transform;
  glm::mat4 inverseTransform;
  glm::mat4 invTranspose;
  int meshid;
};

struct Triangle {
  glm::vec3 vertices[3];
  glm::vec3 normal;

  Triangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3) {
    vertices[0] = v1;
    vertices[1] = v2;
    vertices[2] = v3;
    normal = glm::normalize(glm::cross(v2 - v1, v3 - v1));
  }
};




struct Mesh {
  int triangleStart;
  int triangleEnd;

  glm::vec3 boxMin;
  glm::vec3 boxMax;

  Mesh(int start, int end, std::vector<Triangle> & triangles) {
    triangleStart = start;
    triangleEnd = end;
    for (int i = triangleStart; i < triangleEnd; i++) {
      for (int j = 0; j < 3; j++) {
        boxMin.x = glm::min(boxMin.x, triangles[i].vertices[j].x);
        boxMin.y = glm::min(boxMin.y, triangles[i].vertices[j].y);
        boxMin.z = glm::min(boxMin.z, triangles[i].vertices[j].z);
        boxMax.x = glm::max(boxMax.x, triangles[i].vertices[j].x);
        boxMax.y = glm::max(boxMax.y, triangles[i].vertices[j].y);
        boxMax.z = glm::max(boxMax.z, triangles[i].vertices[j].z);
      }
    }
  }
};

struct Material {
  glm::vec3 color;
  struct {
    float exponent;
    glm::vec3 color;
  } specular;
  float hasReflective;
  float hasRefractive;
  float indexOfRefraction;
  float emittance;
};

struct Camera {
  glm::ivec2 resolution;
  glm::vec3 position;
  glm::vec3 lookAt;
  glm::vec3 view;
  glm::vec3 up;
  glm::vec3 right;
  glm::vec2 fov;
  glm::vec2 pixelLength;
};

struct RenderState {
  Camera camera;
  unsigned int iterations;
  int traceDepth;
  std::vector<glm::vec3> image;
  std::string imageName;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

struct PathSegment {
  Ray ray;
  glm::vec3 color;
  int pixelIndex;
  int remainingBounces;
  bool insideRefractiveObject;
  ShadeableIntersection intersection;
};

