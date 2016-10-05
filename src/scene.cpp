#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

struct SortGeomByType {
  bool operator()(Geom & g1, Geom & g2) {
    return g1.type < g2.type;
  }
};

Scene::Scene(string filename) {
	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		cout << "Error reading from file - aborting!" << endl;
		throw;
	}
	while (fp_in.good()) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
				loadMaterial(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
				loadGeom(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
				loadCamera();
				cout << " " << endl;
			}
      else if (strcmp(tokens[0].c_str(), "MESH") == 0) {
        loadMesh(tokens[1]);
        cout << " " << endl;
      }
		}
	}
}

int Scene::loadMesh(string meshid) {
  int id = atoi(meshid.c_str());
  if (id != meshes.size()) {
    cout << "ERROR: MESH ID does not match expected number of meshes" << endl;
    return -1;
  }
  else {
    cout << "Loading Mesh " << id << "..." << endl;
    string line;
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
      tinyobj::attrib_t attrib;
      vector<tinyobj::shape_t> shapes;
      vector<tinyobj::material_t> materials;
      string err;
      cout << "Loading mesh file: " << line << endl;
      bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, line.c_str());
      if (!err.empty()) {
        cout << "Error: " << err << endl;
      }
      if (!ret) {
        cout << "Could not load!" << endl;
      }
      int startCount = triangles.size();

      for (int i = 0; i < shapes.size(); i++) {
        int index_offset = 0;
        for (int j = 0; j < shapes[i].mesh.num_face_vertices.size(); j++) {
          int fv = shapes[i].mesh.num_face_vertices[j];
          if (fv != 3) {
            cout << "Non-triangular face!" << endl;
            index_offset += fv;
            continue;
          }
          glm::vec3 verts[3];
          for (int k = 0; k < 3; k++) {
            tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + k];
            verts[k] = glm::vec3(
              attrib.vertices[3 * idx.vertex_index + 0],
              attrib.vertices[3 * idx.vertex_index + 1],
              attrib.vertices[3 * idx.vertex_index + 2]
            );
          }
          index_offset += fv;
          triangles.push_back(Triangle(verts[0], verts[1], verts[2]));
        }
      }
      int endCount = triangles.size();
      meshes.push_back(Mesh(startCount, endCount, triangles, triangles));
    }
    else {
      cout << "Could not read mesh obj file!" << endl;
      return -1;
    }
  }
}

int Scene::loadGeom(string objectid) {
	int id = atoi(objectid.c_str());
	if (id != geoms.size()) {
		cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
		return -1;
	}
	else {
		cout << "Loading Geom " << id << "..." << endl;
		Geom newGeom;
		string line;

		//load object type
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			if (strcmp(line.c_str(), "sphere") == 0) {
				cout << "Creating new sphere..." << endl;
				newGeom.type = SPHERE;
			}
			else if (strcmp(line.c_str(), "cube") == 0) {
				cout << "Creating new cube..." << endl;
				newGeom.type = CUBE;
			}
			else if (strcmp(line.c_str(), "mesh") == 0) {
				cout << "Creating new mesh..." << endl;
				newGeom.type = MESH;
			}
		}

		// load mesh ID
		if (newGeom.type == MESH) {
			utilityCore::safeGetline(fp_in, line);
			if (!line.empty() && fp_in.good()) {
				vector<string> tokens = utilityCore::tokenizeString(line);
				newGeom.meshid = atoi(tokens[1].c_str());
				cout << "Connecting Geom " << objectid << " to Mesh " << newGeom.meshid << "..." << endl;
			}
		}

		//link material
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			newGeom.materialid = atoi(tokens[1].c_str());
			cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
		}

		//load transformations
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);

			//load tranformations
			if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}

			utilityCore::safeGetline(fp_in, line);
		}

		newGeom.transform = utilityCore::buildTransformationMatrix(
			newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    if (newGeom.type == MESH) {
      meshGeoms.push_back(newGeom);
    }
    else {
      geoms.push_back(newGeom);
    }
		return 1;
	}
}

int Scene::loadCamera() {
	cout << "Loading Camera ..." << endl;
	RenderState &state = this->state;
	Camera &camera = state.camera;
	float fovy;

	//load static properties
	for (int i = 0; i < 5; i++) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "RES") == 0) {
			camera.resolution.x = atoi(tokens[1].c_str());
			camera.resolution.y = atoi(tokens[2].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
			fovy = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
			state.iterations = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
			state.traceDepth = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
			state.imageName = tokens[1];
		}
	}

	string line;
	utilityCore::safeGetline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "EYE") == 0) {
			camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
			camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "UP") == 0) {
			camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}

		utilityCore::safeGetline(fp_in, line);
	}

	//calculate fov based on resolution
	float yscaled = tan(fovy * (PI / 180));
	float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
	float fovx = (atan(xscaled) * 180) / PI;
	camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
		, 2 * yscaled / (float)camera.resolution.y);

	camera.view = glm::normalize(camera.lookAt - camera.position);

	//set up render camera stuff
	int arraylen = camera.resolution.x * camera.resolution.y;
	state.image.resize(arraylen);
	std::fill(state.image.begin(), state.image.end(), glm::vec3());

	cout << "Loaded camera!" << endl;
	return 1;
}

int Scene::loadMaterial(string materialid) {
	int id = atoi(materialid.c_str());
	if (id != materials.size()) {
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}
	else {
		cout << "Loading Material " << id << "..." << endl;
		Material newMaterial;

		//load static properties
		for (int i = 0; i < 7; i++) {
			string line;
			utilityCore::safeGetline(fp_in, line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "RGB") == 0) {
				glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.color = color;
			}
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
				newMaterial.specular.exponent = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
				glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.specular.color = specColor;
			}
			else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
				newMaterial.emittance = atof(tokens[1].c_str());
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}
