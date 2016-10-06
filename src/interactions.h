#pragma once

#include "intersections.h"

#include <glm/gtx/rotate_vector.hpp>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 calculateRandomDirectionWithSpecular(float specularExponent, glm::vec3 direction,
glm::vec3 normal, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float theta = acos(pow(u01(rng), 1.0f / (1.0f + specularExponent)));
	float phi = 2 * PI * u01(rng);

	glm::vec3 randDir(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

	glm::vec3 newUp = glm::normalize(glm::reflect(direction, normal));
	glm::vec3 oldUp(0, 0, 1);

	glm::vec3 rotationAxis = glm::cross(oldUp, newUp);
	float rotationAngle = acos(glm::dot(newUp, oldUp));

	return glm::rotate(randDir, rotationAngle, rotationAxis);
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
	Ray & ray = pathSegment.ray;
  ray.origin = intersect;
	thrust::uniform_real_distribution<float> u01(0, 1);
  const float specularWeight = 0.5f;
  const float specularWeightInverse = (specularWeight > 0.0f) ? 1.0f / specularWeight : 0.0f;
  const float diffuseWeightInverse = 1.0f / (1 - specularWeight);
	// Specular highlight
  if (u01(rng) < specularWeight) {
		ray.direction = calculateRandomDirectionWithSpecular(m.specular.exponent, ray.direction, normal, rng);
		pathSegment.color *= m.specular.color * specularWeightInverse;
  }
	// Diffuse color
  else {
		if (m.hasReflective > 0.0f) {
			ray.direction = glm::reflect(ray.direction, normal);
		}
		else if (m.hasRefractive > 0.0f) {
			float refractionCoeff = (pathSegment.insideRefractiveObject) ? m.indexOfRefraction : (1.0f / m.indexOfRefraction);
      // Schlick's approximation for fresnel
      float R_0_sqrt = (m.indexOfRefraction - 1.0f) / (m.indexOfRefraction + 1.0f);
      float R_0 = R_0_sqrt * R_0_sqrt;
      float R = R_0 + (1.0f - R_0) * glm::pow(1.0f - fabs(glm::dot(normal, ray.direction)), 5.0f);
      if (u01(rng) < R) {
        ray.direction = glm::normalize(glm::reflect(ray.direction, normal));
      }
      else {
        ray.direction = glm::normalize(glm::refract(ray.direction, normal, refractionCoeff));
      }
			if (glm::dot(ray.direction, normal) < 0.0f) {
				pathSegment.insideRefractiveObject = !pathSegment.insideRefractiveObject;
			}
		}
		else {
			ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
			pathSegment.color *= glm::dot(ray.direction, normal);
		}
		pathSegment.color *= m.color * diffuseWeightInverse;
  }
}
