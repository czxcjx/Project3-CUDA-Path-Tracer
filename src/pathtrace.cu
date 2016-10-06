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
#define CALCMESHSEPARATELY 1
#define ANTIALIAS_SAMPLE_SIDE 2


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
static glm::vec3 * dev_final_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static PathSegment * dev_cached_paths = NULL;
static Geom * dev_meshgeoms = NULL;
static Mesh * dev_meshes = NULL;
static Triangle * dev_triangles = NULL;

static glm::ivec3 * dev_path_mesh_intersections = NULL;
static ShadeableIntersection * dev_path_mesh_intersection_dists = NULL;
static glm::ivec3 * dev_pm_intersection_out = NULL;
static ShadeableIntersection * dev_pm_intersection_dists_out = NULL;
int numGeoms = 0;

void pathtraceInit(Scene *scene) {
  hst_scene = scene;
  const Camera &cam = hst_scene->state.camera;
#if ANTIALIAS_SAMPLE_SIDE == 0
  const int pixelcount = cam.resolution.x * cam.resolution.y;
#else
  const int actual_pixelcount = cam.resolution.x * cam.resolution.y;
  const int pixelcount = actual_pixelcount * ANTIALIAS_SAMPLE_SIDE * ANTIALIAS_SAMPLE_SIDE;
#endif

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
#if ANTIALIAS_SAMPLE_SIDE != 0
  cudaMalloc(&dev_final_image, actual_pixelcount * sizeof(glm::vec3));
#else
  dev_final_image = dev_image;
#endif
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

#if CALCMESHSEPARATELY == 1
  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_meshgeoms, scene->meshGeoms.size() * sizeof(Geom));
  cudaMemcpy(dev_meshgeoms, scene->meshGeoms.data(), scene->meshGeoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_path_mesh_intersections, GRID_FULL * pixelcount * scene->meshGeoms.size() * sizeof(glm::ivec3));
  cudaMalloc(&dev_path_mesh_intersection_dists, GRID_FULL * pixelcount * scene->meshGeoms.size() * sizeof(ShadeableIntersection));
  cudaMalloc(&dev_pm_intersection_out, pixelcount * sizeof(glm::ivec3));
  cudaMalloc(&dev_pm_intersection_dists_out, pixelcount * sizeof(ShadeableIntersection));
  numGeoms = scene->geoms.size();
#else
  numGeoms = scene->geoms.size() + scene->meshGeoms.size();
  cudaMalloc(&dev_geoms, numGeoms * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_geoms + scene->geoms.size(), scene->meshGeoms.data(), scene->meshGeoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif


  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  // TODO: initialize any extra device memeory you need
  cudaMalloc(&dev_cached_paths, pixelcount * sizeof(PathSegment));
  cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(Mesh));
  cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
  cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
  cudaFree(dev_image);  // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  // TODO: clean up any extra device memory you created
  cudaFree(dev_cached_paths);
  cudaFree(dev_meshes);
  cudaFree(dev_triangles);
#if CALCMESHSEPARATELY == 1
  cudaFree(dev_meshgeoms);
  cudaFree(dev_path_mesh_intersections);
  cudaFree(dev_path_mesh_intersection_dists);
  cudaFree(dev_pm_intersection_out);
  cudaFree(dev_pm_intersection_dists_out);
#endif

#if ANTIALIAS_SAMPLE_SIDE != 0
  cudaFree(dev_final_image);
#endif

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
#if ANTIALIAS_SAMPLE_SIDE == 0
    int index = x + (y * cam.resolution.x);
    PathSegment & segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

    segment.ray.direction = glm::normalize(cam.view
      - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
      - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
      );

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
    segment.insideRefractiveObject = false;
#else
    float subpixel_side = 1.0f / (float)ANTIALIAS_SAMPLE_SIDE;
    thrust::default_random_engine rng = makeSeededRandomEngine(iter,  x + y * cam.resolution.x, traceDepth);
    thrust::uniform_real_distribution<float> u(0, subpixel_side);
    for (int i = 0; i < ANTIALIAS_SAMPLE_SIDE; i++) {
      for (int j = 0; j < ANTIALIAS_SAMPLE_SIDE; j++) {
        int index = (x + (y * cam.resolution.x)) * ANTIALIAS_SAMPLE_SIDE * ANTIALIAS_SAMPLE_SIDE + ANTIALIAS_SAMPLE_SIDE * j + i;
        PathSegment & segment = pathSegments[index];
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        segment.ray.direction = glm::normalize(cam.view 
          - cam.right * cam.pixelLength.x * ((float)x - 0.5f + i * subpixel_side + u(rng) - (float)cam.resolution.x * 0.5f)
          - cam.up * cam.pixelLength.y * ((float)y - 0.5f + j * subpixel_side + u(rng) - (float)cam.resolution.y * 0.5f)
          );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.insideRefractiveObject = false;
      }
    }
#endif
  }
}

__global__ void pathTraceSphereBox(
  int depth
  , int num_paths
  , PathSegment * pathSegments
  , Geom * geoms
  , int geoms_size
  , Mesh * meshes
  , Triangle * triangles)
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
#if CALCMESHSEPARATELY == 0
      else if (geom.type == MESH)
      {
        t = meshIntersectionTest(geom, meshes, triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside);
      }
#endif
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

// Color using only the parts with no bounces
__global__ void kernAntialiasGather(int pixelcount, glm::vec3 * final_image, glm::vec3 * image) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < pixelcount)
  {
    int subpixel_count = ANTIALIAS_SAMPLE_SIDE * ANTIALIAS_SAMPLE_SIDE;
    int base_index = index * subpixel_count;
    glm::vec3 sumColors(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < subpixel_count; i++) {
      sumColors += image[base_index + i];
    }
    final_image[index] = sumColors / (float)subpixel_count;
  }
}


__global__ void kernCalculateMeshBoundingBoxIntersections(int nPaths, int nMeshes, PathSegment * iterationPaths,
  Geom * meshgeoms, Mesh * meshes, glm::ivec3 * intersections) {
  int pathIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  int meshIndex = (blockIdx.y * blockDim.y) + threadIdx.y;
  int gridIndex = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (pathIndex < nPaths && meshIndex < nMeshes && gridIndex < GRID_FULL) {
    int idx = pathIndex * nMeshes * GRID_FULL + meshIndex * GRID_FULL + gridIndex;
    Geom & meshGeom = meshgeoms[meshIndex];
    Mesh & mesh = meshes[meshGeom.meshid];
    PathSegment & path = iterationPaths[pathIndex];
    Ray r = path.ray;

    glm::vec3 gridSize = (mesh.boxMax - mesh.boxMin) / (float)GRID_SIZE;
    glm::vec3 gridMin(
      gridSize.x * (gridIndex % GRID_SIZE),
      gridSize.y * ((gridIndex / GRID_SIZE) % GRID_SIZE),
      gridSize.z * (gridIndex / (GRID_SIZE * GRID_SIZE)));
    gridMin += mesh.boxMin;
    glm::vec3 gridMax = gridMin + gridSize;

    glm::vec3 ro = multiplyMV(meshGeom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(meshGeom.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float tmin, tmax;
    glm::vec3 tmin_n, tmax_n;
    if (orientedBoxIntersection(gridMin, gridMax, rt, tmin, tmax, tmin_n, tmax_n)) {
      intersections[idx] = glm::ivec3(pathIndex, meshIndex, gridIndex);
    }
    else {
      intersections[idx] = glm::ivec3(-1, -1, -1);
    }
  }
}

__global__ void kernPathTraceMesh(int nIntersections, PathSegment * iterationPaths,
  Geom * meshgeoms, Mesh * meshes, glm::ivec3 * intersections, Triangle * triangles,
  ShadeableIntersection * intersection_out) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < nIntersections) {
    glm::vec3 tmp_intersect;
    glm::ivec3 idx = intersections[index];
    bool outside;
    intersection_out[index].t = meshIntersectionTest(meshgeoms[idx.y], meshes,
      triangles, iterationPaths[idx.x].ray, 
      tmp_intersect, intersection_out[index].surfaceNormal, outside, idx.z, 
      iterationPaths[idx.x].insideRefractiveObject);
    intersection_out[index].materialId = meshgeoms[idx.y].materialid;
  }
}

__global__ void kernTakeMeshIntersection(int nIntersections, PathSegment * iterationPaths,
  glm::ivec3 * intersectionKeys, ShadeableIntersection * intersectionValues) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < nIntersections) {
    ShadeableIntersection & meshIntersection = intersectionValues[index];
    if (meshIntersection.t > EPSILON) {
      int pathIndex = intersectionKeys[index].x;
      ShadeableIntersection & intersection = iterationPaths[pathIndex].intersection;
      ShadeableIntersection & meshIntersection = intersectionValues[index];
      if (intersection.t < EPSILON || intersection.t > meshIntersection.t) {
        intersection = meshIntersection;
      }
    }
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

struct IsNonintersection {
  __host__ __device__ bool operator() (const glm::ivec3 v) {
    return v.x == -1 && v.y == -1 && v.z == -1;
  }
};

struct TakeMinIntersection {
  __host__ __device__ ShadeableIntersection operator() (ShadeableIntersection i1, ShadeableIntersection i2) {
    if (i1.t < EPSILON) {
      return i2;
    }
    if (i2.t < EPSILON) {
      return i1;
    }
    return (i1.t < i2.t) ? i1 : i2;
  }
};

struct ComparePathKey {
  __host__ __device__ bool operator() (const glm::ivec3 v1, const glm::ivec3 v2) {
    return v1.x == v2.x;
  }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
  const int traceDepth = hst_scene->state.traceDepth;
  const Camera &cam = hst_scene->state.camera;
#if ANTIALIAS_SAMPLE_SIDE == 0
  const int pixelcount = cam.resolution.x * cam.resolution.y;
  const int actual_pixelcount = pixelcount;
#else
  const int actual_pixelcount = cam.resolution.x * cam.resolution.y;
  const int pixelcount = actual_pixelcount * ANTIALIAS_SAMPLE_SIDE * ANTIALIAS_SAMPLE_SIDE;
#endif

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
#if (CACHEFIRSTBOUNCE == 1 && ANTIALIAS_SAMPLE_SIDE == 0)
  if (iter > 1) {
    cudaMemcpy(dev_paths, dev_cached_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
  }
  else {
#endif
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
#if (CACHEFIRSTBOUNCE == 1 && ANTIALIAS_SAMPLE_SIDE == 0)
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
#if (CACHEFIRSTBOUNCE == 1 && ANTIALIAS_SAMPLE_SIDE == 0)
    if (iter > 1 || depth > 0) {
#endif

      // spheres and boxes
      pathTraceSphereBox << <numblocksPathSegmentTracing, blockSize1d >> > (
        depth
        , num_pathsInFlight
        , dev_paths
        , dev_geoms
        , numGeoms
        , dev_meshes
        , dev_triangles
        );
      checkCUDAError("trace one bounce");
      cudaDeviceSynchronize();
#if CALCMESHSEPARATELY == 1
      // meshes
      dim3 blockSize3d(8, 8, 8);
      if (hst_scene->meshGeoms.size() > 0) {
        dim3 meshBoxTracing(
          (num_pathsInFlight + blockSize3d.x - 1) / blockSize3d.x,
          (hst_scene->meshGeoms.size() + blockSize3d.y - 1) / blockSize3d.y,
          (GRID_FULL + blockSize3d.z - 1) / blockSize3d.z
          );
        kernCalculateMeshBoundingBoxIntersections << <meshBoxTracing, blockSize3d >> >(
          num_pathsInFlight, hst_scene->meshGeoms.size(), dev_paths, dev_meshgeoms, dev_meshes, dev_path_mesh_intersections);
        checkCUDAError("calculate bounding box intersections");
        cudaDeviceSynchronize();
        thrust::device_ptr<glm::ivec3> dev_thrust_path_mesh_intersections = thrust::device_pointer_cast(dev_path_mesh_intersections);
        int numIntersections = num_pathsInFlight * hst_scene->meshGeoms.size() * GRID_FULL;
        thrust::device_ptr<glm::ivec3> dev_thrust_path_mesh_intersections_end =
          thrust::remove_if(dev_thrust_path_mesh_intersections, dev_thrust_path_mesh_intersections + numIntersections, IsNonintersection());
        int numActualIntersections = dev_thrust_path_mesh_intersections_end - dev_thrust_path_mesh_intersections;
        printf("Culled from %d to %d intersections\n", numIntersections, numActualIntersections);
        dim3 meshTracing((numActualIntersections + blockSize1d - 1) / blockSize1d);
        kernPathTraceMesh << <meshTracing, blockSize1d >> > (
          numActualIntersections, dev_paths, dev_meshgeoms, dev_meshes, dev_path_mesh_intersections, dev_triangles,
          dev_path_mesh_intersection_dists);
        checkCUDAError("calculate ray-mesh intersections");
        cudaDeviceSynchronize();

        thrust::device_ptr<ShadeableIntersection> dev_thrust_path_mesh_intersection_dists =
          thrust::device_pointer_cast(dev_path_mesh_intersection_dists);
        thrust::device_ptr<glm::ivec3> dev_thrust_pm_intersection_out =
          thrust::device_pointer_cast(dev_pm_intersection_out);
        thrust::device_ptr<ShadeableIntersection> dev_thrust_pm_intersection_dists_out =
          thrust::device_pointer_cast(dev_pm_intersection_dists_out);

        thrust::pair<thrust::device_ptr<glm::ivec3>, thrust::device_ptr<ShadeableIntersection>> reductionResult =
          thrust::reduce_by_key(dev_thrust_path_mesh_intersections,
          dev_thrust_path_mesh_intersections + numActualIntersections,
          dev_thrust_path_mesh_intersection_dists,
          dev_thrust_pm_intersection_out,
          dev_thrust_pm_intersection_dists_out,
          ComparePathKey(),
          TakeMinIntersection());
        int numPathIntersections = reductionResult.first - dev_thrust_pm_intersection_out;
        kernTakeMeshIntersection << <dim3(numPathIntersections + blockSize1d - 1 / blockSize1d), blockSize1d >> >(
          numPathIntersections, dev_paths, dev_pm_intersection_out, dev_pm_intersection_dists_out);
        checkCUDAError("get new ray-mesh intersections");
        cudaDeviceSynchronize();
      }
#endif
      
#if (CACHEFIRSTBOUNCE == 1 && ANTIALIAS_SAMPLE_SIDE == 0)
    }
#endif

#if (CACHEFIRSTBOUNCE == 1 && ANTIALIAS_SAMPLE_SIDE == 0)
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
#if ANTIALIAS_SAMPLE_SIDE != 0
  dim3 numBlocksPixels = (actual_pixelcount + blockSize1d - 1) / blockSize1d;
  kernAntialiasGather << <numBlocksPixels, blockSize1d >> >(actual_pixelcount, dev_final_image, dev_image);
#endif


  ///////////////////////////////////////////////////////////////////////////

  // Send results to OpenGL buffer for rendering
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_final_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_final_image,
    actual_pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}
