#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define GRID_SIZE 4

enum GeomType {
  SPHERE = 0,
  CUBE = 1,
  MESH = 2,
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

  struct {
    int start, end;
  } gridIdx[GRID_SIZE][GRID_SIZE][GRID_SIZE];

  Mesh(int start, int end, std::vector<Triangle> & triangles, std::vector<Triangle> & gridTriangles) {
    triangleStart = start;
    triangleEnd = end;

    std::vector<Triangle> grid[GRID_SIZE][GRID_SIZE][GRID_SIZE];

    boxMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    boxMax = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);

    for (int i = triangleStart; i < triangleEnd; i++) {
      for (int j = 0; j < 3; j++) {
        boxMin = utilityCore::vecMin(boxMin, triangles[i].vertices[j]);
        boxMax = utilityCore::vecMax(boxMax, triangles[i].vertices[j]);
      }
    }

    glm::vec3 boxDim = boxMax - boxMin;

    for (int i = triangleStart; i < triangleEnd; i++) {
      glm::vec3 triangleMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
      glm::vec3 triangleMax = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);
      for (int j = 0; j < 3; j++) {
        triangleMin = utilityCore::vecMin(triangleMin, triangles[i].vertices[j]);
        triangleMax = utilityCore::vecMax(triangleMax, triangles[i].vertices[j]);
      }
      for (int x = 0; x < GRID_SIZE; x++) {
        for (int y = 0; y < GRID_SIZE; y++) {
          for (int z = 0; z < GRID_SIZE; z++) {

            glm::vec3 gridMin = boxMin + glm::vec3(x, y, z) * (boxDim / (float)GRID_SIZE);
            glm::vec3 gridMax = boxMin + glm::vec3(x + 1, y + 1, z + 1) * (boxDim / (float)GRID_SIZE);

            if (utilityCore::aabbIntersect(triangleMin, triangleMax, gridMin, gridMax)) {
              grid[x][y][z].push_back(triangles[i]);
            }
          }
        }
      }
    }
    for (int x = 0; x < GRID_SIZE; x++) {
      for (int y = 0; y < GRID_SIZE; y++) {
        for (int z = 0; z < GRID_SIZE; z++) {
          gridIdx[x][y][z].start = gridTriangles.size();
          for (int i = 0; i < grid[x][y][z].size(); i++) {
            gridTriangles.push_back(grid[x][y][z][i]);
          }
          gridIdx[x][y][z].end = gridTriangles.size();
        }
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