#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define GROUPBYMAT 0
#define CACHEFIRSTBOUNCE 1


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err) {
    return;
  }

  fprintf(stderr, "CUDA error");
  if (file) {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
  getchar();
#  endif
  exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
  int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
  return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
  int iter, glm::vec3* image) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    glm::vec3 pix = image[index];

    glm::ivec3 color;
    color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
    color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
    color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 0;
    pbo[index].x = color.x;
    pbo[index].y = color.y;
    pbo[index].z = color.z;
  }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static PathSegment * dev_cached_paths = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
  hst_scene = scene;
  const Camera &cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_cached_paths, pixelcount * sizeof(PathSegment));

  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  // TODO: initialize any extra device memeory you need

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
  cudaFree(dev_image);  // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  // TODO: clean up any extra device memory you created
	cudaFree(dev_cached_paths);

  checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {
    int index = x + (y * cam.resolution.x);
    PathSegment & segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

    // TODO: implement antialiasing by jittering the ray
    segment.ray.direction = glm::normalize(cam.view
      - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
	  - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
      );

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
		segment.insideRefractiveObject = false;
  }
}

__global__ void pathTraceOneBounce(
  int depth
  , int num_paths
  , PathSegment * pathSegments
  , Geom * geoms
  , int geoms_size)
{
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pathSegments[path_index].remainingBounces == 0) {
    return;
  }

  if (path_index < num_paths)
  {
    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
      Geom & geom = geoms[i];

      if (geom.type == CUBE)
      {
        t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
      }
      else if (geom.type == SPHERE)
      {
        t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
      }
      // TODO: add more intersection tests here... triangle? metaball? CSG?
      // Compute the minimum t from the intersection tests to determine what
      // scene geometry object was hit first.
      if (t > 1e-3 && t_min > t)
      {
        t_min = t;
        hit_geom_index = i;
        intersect_point = tmp_intersect;
        normal = tmp_normal;
      }
    }

    if (hit_geom_index == -1)
    {
		pathSegments[path_index].intersection.t = -1.0f;
    }
    else
    {
      //The ray hits something
			pathSegments[path_index].intersection.t = t_min;
			pathSegments[path_index].intersection.materialId = geoms[hit_geom_index].materialid;
			pathSegments[path_index].intersection.surfaceNormal = normal;
    }
  }
}

__global__ void simpleBSDFShader(int iter, int num_paths, PathSegment * pathSegments, Material * materials) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_paths) {
    return;
  }
  ShadeableIntersection & intersection = pathSegments[idx].intersection;
  if (pathSegments[idx].remainingBounces >= 0) {
    // If intersection exists
    if (intersection.t > 0.0f) {

      Material material = materials[intersection.materialId];
      // Hit a light
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= material.emittance * material.color;
        pathSegments[idx].remainingBounces = -1;
      }
      // Bouncing off a nonlight
      else {
        scatterRay(
          pathSegments[idx],
          pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction,
          intersection.surfaceNormal,
          material,
          makeSeededRandomEngine(iter, idx, 0)
          );
        pathSegments[idx].remainingBounces--;
        if (pathSegments[idx].remainingBounces == -1) {
          pathSegments[idx].color = glm::vec3(0.0f);
        }
      }
    }
    else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = -1;
    }
  }
}

// Color using only the parts with no bounces
__global__ void partialGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths)
  {
    PathSegment iterationPath = iterationPaths[index];
    if (iterationPath.remainingBounces < 0) {
      image[iterationPath.pixelIndex] += iterationPath.color;
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths)
  {
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
  }
}

struct HasNoBounces {
  __host__ __device__ bool operator() (const PathSegment & path) {
    return path.remainingBounces < 0;
  }
};

struct SortByMaterial {
	__host__ __device__ bool operator() (const PathSegment & p1, const PathSegment & p2) {
		return p1.intersection.materialId < p2.intersection.materialId;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
  const int traceDepth = hst_scene->state.traceDepth;
  const Camera &cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  // 2D block for generating ray from camera
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // 1D block for path tracing
  const int blockSize1d = 128;

  ///////////////////////////////////////////////////////////////////////////

  // Recap:
  // * Initialize array of path rays (using rays that come out of the camera)
  //   * You can pass the Camera object to that kernel.
  //   * Each path ray must carry at minimum a (ray, color) pair,
  //   * where color starts as the multiplicative identity, white = (1, 1, 1).
  //   * This has already been done for you.
  // * For each depth:
  //   * Compute an intersection in the scene for each path ray.
  //     A very naive version of this has been implemented for you, but feel
  //     free to add more primitives and/or a better algorithm.
  //     Currently, intersection distance is recorded as a parametric distance,
  //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
  //     * Color is attenuated (multiplied) by reflections off of any object
  //   * TODO: Stream compact away all of the terminated paths.
  //     You may use either your implementation or `thrust::remove_if` or its
  //     cousins.
  //     * Note that you can't really use a 2D kernel launch any more - switch
  //       to 1D.
  //   * TODO: Shade the rays that intersected something or didn't bottom out.
  //     That is, color the ray by performing a color computation according
  //     to the shader, then generate a new ray to continue the ray path.
  //     We recommend just updating the ray's PathSegment in place.
  //     Note that this step may come before or after stream compaction,
  //     since some shaders you write may also cause a path to terminate.
  // * Finally, add this iteration's results to the image. This has been done
  //   for you.

  // 
	int depth = 0;
#if CACHEFIRSTBOUNCE == 1
	if (iter > 1) {
		cudaMemcpy(dev_paths, dev_cached_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	else {
#endif
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
#if CACHEFIRSTBOUNCE == 1
	}
#endif

  
  PathSegment* dev_path_end = dev_paths + pixelcount;
  int num_paths = dev_path_end - dev_paths;
  int num_pathsInFlight = num_paths;

	thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
  while (!iterationComplete) {
		dim3 numblocksPathSegmentTracing = (num_pathsInFlight + blockSize1d - 1) / blockSize1d;

    // tracing
#if CACHEFIRSTBOUNCE == 1
		if (iter > 1 || depth > 0) {
#endif
			pathTraceOneBounce << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_pathsInFlight
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
#if CACHEFIRSTBOUNCE == 1
		}
#endif

#if CACHEFIRSTBOUNCE == 1
		// Cache first bounce
		if (iter == 1 && depth == 0) {
			cudaMemcpy(dev_cached_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		}
#endif
    depth++;

    // TODO:
    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.
#if GROUPBYMAT == 1
	thrust::sort(dev_thrust_paths, dev_thrust_paths + num_pathsInFlight, SortByMaterial());
#endif

    simpleBSDFShader << <numblocksPathSegmentTracing, blockSize1d >> > (
      iter,
      num_pathsInFlight,
      dev_paths,
      dev_materials
      );

    dim3 numBlocksPixels = (num_pathsInFlight + blockSize1d - 1) / blockSize1d;
    partialGather << <numBlocksPixels, blockSize1d >> >(num_pathsInFlight, dev_image, dev_paths);

    thrust::device_ptr<PathSegment> dev_thrust_newPathEnd;
    dev_thrust_newPathEnd = thrust::remove_if(
      thrust::device,
      dev_thrust_paths,
      dev_thrust_paths + num_pathsInFlight,
      HasNoBounces());
    num_pathsInFlight = dev_thrust_newPathEnd - dev_thrust_paths;

    iterationComplete = depth >= traceDepth || num_pathsInFlight == 0;
  }

  // Assemble this iteration and apply it to the image
  //dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
  //finalGather << <numBlocksPixels, blockSize1d >> >(num_paths_in_flight, dev_image, dev_paths);
  

  ///////////////////////////////////////////////////////////////////////////

  // Send results to OpenGL buffer for rendering
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
    pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}
