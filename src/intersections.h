#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT

__host__ __device__ bool orientedBoxIntersection(glm::vec3 boxMin, glm::vec3 boxMax, Ray r,
  float &tmin, float &tmax, glm::vec3 & tmin_n, glm::vec3 & tmax_n) {
  tmin = -1e38f;
  tmax = 1e38f;
  for (int xyz = 0; xyz < 3; ++xyz) {
    float qdxyz = r.direction[xyz];
    /*if (glm::abs(qdxyz) > 0.00001f)*/ {
      float t1 = (boxMin[xyz] - r.origin[xyz]) / qdxyz;
      float t2 = (boxMax[xyz] - r.origin[xyz]) / qdxyz;
      float ta = glm::min(t1, t2);
      float tb = glm::max(t1, t2);
      glm::vec3 n;
      n[xyz] = t2 < t1 ? +1 : -1;
      if (ta > 0 && ta > tmin) {
        tmin = ta;
        tmin_n = n;
      }
      if (tb < tmax) {
        tmax = tb;
        tmax_n = n;
      }
    }
  }
  return tmax >= tmin && tmax > 0;
}

/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    orientedBoxIntersection(glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, 0.5f, 0.5f), q, tmin, tmax, tmin_n, tmax_n);

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

/**
* Test intersection between a ray and a mesh
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float meshIntersectionTest(Geom geom, Mesh * meshes, Triangle * triangles, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {

	glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
	Ray rt;
	rt.origin = ro;
	rt.direction = rd;
	
	Mesh & mesh = meshes[geom.meshid];

  float tmin, tmax;
  glm::vec3 tmin_n, tmax_n;
  if (!orientedBoxIntersection(mesh.boxMin, mesh.boxMax, rt, tmin, tmax, tmin_n, tmax_n)) {
    return -1.0f;
  }

	float t_min = FLT_MAX;
	for (int i = mesh.triangleStart; i < mesh.triangleEnd; i++) {
    glm::vec3 result;
    bool hasIntersect = glm::intersectRayTriangle(rt.origin, rt.direction, triangles[i].vertices[0], triangles[i].vertices[1], triangles[i].vertices[2], result);
    glm::vec3 tmp_intersect = getPointOnRay(rt, result.z);
    float t = glm::length(rt.origin - tmp_intersect);
		if (t > 1e-3 && t_min > t) {
			t_min = t;
			intersectionPoint = multiplyMV(geom.transform, glm::vec4(tmp_intersect, 1.0f));
			normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(triangles[i].normal, 0.0f)));
			outside = glm::dot(rt.direction, normal) < 0.0f;
		}
	}

  /*glm::vec3 gridDim = (mesh.boxMax - mesh.boxMin) / (float)GRID_SIZE;

  for (int x = 0; x < GRID_SIZE; x++) {
    for (int y = 0; y < GRID_SIZE; y++) {
      for (int z = 0; z < GRID_SIZE; z++) {
        glm::vec3 gridMin = mesh.boxMin + glm::vec3(x, y, z) * gridDim;
        if (orientedBoxIntersection(gridMin, gridMin + gridDim, rt, tmin, tmax, tmin_n, tmax_n)) {
          for (int i = mesh.gridIdx[x][y][z].start; i < mesh.gridIdx[x][y][z].end; i++) {
            glm::vec3 result;
            bool hasIntersect = glm::intersectRayTriangle(rt.origin, rt.direction, triangles[i].vertices[0], triangles[i].vertices[1], triangles[i].vertices[2], result);
            glm::vec3 tmp_intersect = getPointOnRay(rt, result.z);
            float t = glm::length(rt.origin - tmp_intersect);
            if (t > 1e-3 && t_min > t) {
              t_min = t;
              intersectionPoint = multiplyMV(geom.transform, glm::vec4(tmp_intersect, 1.0f));
              normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(triangles[i].normal, 0.0f)));
              outside = glm::dot(rt.direction, normal) < 0.0f;
            }
          }
        }
      }
    }
  }*/
	if (t_min == FLT_MAX) {
		return -1.0f;
	}
	else {
		return glm::length(r.origin - intersectionPoint);
	}
}
