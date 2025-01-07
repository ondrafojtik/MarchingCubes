#define _CRT_SECURE_NO_WARNINGS

#define M_PI       3.14159265358979323846   // pi

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

typedef cv::flann::GenericIndex<cv::flann::L2<float>> SearchIndex;
typedef std::vector<int> Neighbourhood;

#include "vector3.h"
#include <array>
#include <unordered_map>

#include "half.hpp"
using half_float::half;

#include "table.h"

const size_t N = 40 * 40 * 40;

//const float step = 1.0f / 20.0f;
//const float step = 1.0f / 40.0f;
const float step = 1.0f / 100.0f;

struct Particle
{
	half position_x; // particle position (m)
	half position_y;
	half position_z;

	half velocity_x; // particle velocity (m/s)
	half velocity_y;
	half velocity_z;

	half rho; // density (kg/m3)
	half pressure;
	half radius; // particle radius (m)

	Vector3 get_position() { return Vector3(position_x, position_y, position_z); }
	std::array<float, 3> get_position_() { return { position_x, position_y, position_z }; }

	float get_compute_value() 
	{
		return (float)((4.0f / 3.0f) * M_PI * (pow(radius, 3))) * rho;
	}



};

struct Triangle 
{
	std::array<Vector3, 3> vertices;
};

float gaussian(float& r, float& h)
{
	float w = std::cbrt(M_PI) * std::pow(h, 3);
	float w_ = pow((r / h), 2);
	
	float result_ = w * std::exp(-w_);
	return result_;
}

float cubic_spline(float& r, float& h)
{
	float q = r / h;
	float result_ = 0;

	if (0 <= q && q < 1)
		result_ = (2.0f / 3.0f) - pow(q, 2) + (0.5f * pow(q, 3));
	if (1 <= q && q < 2)
		result_ = (1.0f / 6.0f) * pow((2 - q), 3);
	if (q >= 2)
		result_ = 0;

	return (3.0f / (2 * M_PI * pow(h, 3)) * result_);

}

namespace VD
{
	struct cube
	{
		Vector3 position; // center point
		std::vector<Vector3> points;
		std::vector<float> values;

		const float isolevel = 500.0f;

		cube(cv::Point3d p)
		{
			position.x = p.x;
			position.y = p.y;
			position.z = p.z;

			// calculate points positions

			Vector3 p0(position.x - (step/2), position.y - (step/2), position.z - (step/2));
			Vector3 p1(position.x - (step/2), position.y + (step/2), position.z - (step/2));
			Vector3 p2(position.x + (step/2), position.y + (step/2), position.z - (step/2));
			Vector3 p3(position.x + (step/2), position.y - (step/2), position.z - (step/2));
			Vector3 p4(position.x - (step/2), position.y - (step/2), position.z + (step/2));
			Vector3 p5(position.x - (step/2), position.y + (step/2), position.z + (step/2));
			Vector3 p6(position.x + (step/2), position.y + (step/2), position.z + (step/2));
			Vector3 p7(position.x + (step/2), position.y - (step/2), position.z + (step/2));

			points.push_back(p0);
			points.push_back(p1);
			points.push_back(p2);
			points.push_back(p3);
			points.push_back(p4);
			points.push_back(p5);
			points.push_back(p6);
			points.push_back(p7);

			values.push_back(0);
			values.push_back(0);
			values.push_back(0);
			values.push_back(0);
			values.push_back(0);
			values.push_back(0);
			values.push_back(0);
			values.push_back(0);
		}

		Vector3 interpolate2(float isolevel, Vector3& p1, Vector3& p2, float valp1, float valp2) {
			if (abs(isolevel - valp1) < 0.00001) return p1;
			if (abs(isolevel - valp1) < 0.00001) return p2;
			if (abs(valp1 - valp2) < 0.00001) return p1;

			float mu = (isolevel - valp1) / (valp2 - valp1);
			Vector3 p{};
			p.x = p1.x + mu * (p2.x - p1.x);
			p.y = p1.y + mu * (p2.y - p1.y);
			p.z = p1.z + mu * (p2.z - p1.z);
			return p;

		}

		Vector3 interpolate1(float isolevel, Vector3& p1, Vector3& p2, float valp1, float valp2) 
		{
			float value = abs(valp1 - valp2);
			
			if (p2.z < p1.z)
			{
				Vector3 tmp{};
				tmp = p1;
				p1 = p2;
				p2 = tmp;
			}

			Vector3 p{};
			if (abs(valp1 - valp2) > 0.00001)
				p = p1 + (p2 - p1) / (valp2 - valp1) * (value - valp1);
			else
				p = p1;
			return p;

		}

		Vector3 interpolate(float isolevel, Vector3& xi, Vector3& xj, float fi, float fj)
		{
			float df = (isolevel - fi) / (fj - fi);
			Vector3 xiso{};
			xiso = xi + (xj - xi) * df;
			return xiso;
		}

		void make_triangles(std::vector<Triangle>& triangles)
		{
			int cubeindex = 0;

			cubeindex = 0;
			if (values[0] < isolevel) cubeindex |= 1;
			if (values[1] < isolevel) cubeindex |= 2;
			if (values[2] < isolevel) cubeindex |= 4;
			if (values[3] < isolevel) cubeindex |= 8;
			if (values[4] < isolevel) cubeindex |= 16;
			if (values[5] < isolevel) cubeindex |= 32;
			if (values[6] < isolevel) cubeindex |= 64;
			if (values[7] < isolevel) cubeindex |= 128;
			
			
			
			std::vector<Vector3> vertlist(12);
			// Find the vertices where the surface intersects the cube 
			if (edgeTable[cubeindex] & 1)
				vertlist[0] =
				interpolate(isolevel, points[0], points[1], values[0], values[1]);
			if (edgeTable[cubeindex] & 2)
				vertlist[1] =
				interpolate(isolevel, points[1], points[2], values[1], values[2]);
			if (edgeTable[cubeindex] & 4)
				vertlist[2] =
				interpolate(isolevel, points[2], points[3], values[2], values[3]);
			if (edgeTable[cubeindex] & 8)
				vertlist[3] =
				interpolate(isolevel, points[3], points[0], values[3], values[0]);
			if (edgeTable[cubeindex] & 16)
				vertlist[4] =
				interpolate(isolevel, points[4], points[5], values[4], values[5]);
			if (edgeTable[cubeindex] & 32)
				vertlist[5] =
				interpolate(isolevel, points[5], points[6], values[5], values[6]);
			if (edgeTable[cubeindex] & 64)
				vertlist[6] =
				interpolate(isolevel, points[6], points[7], values[6], values[7]);
			if (edgeTable[cubeindex] & 128)
				vertlist[7] =
				interpolate(isolevel, points[7], points[4], values[7], values[4]);
			if (edgeTable[cubeindex] & 256)
				vertlist[8] =
				interpolate(isolevel, points[0], points[4], values[0], values[4]);
			if (edgeTable[cubeindex] & 512)
				vertlist[9] =
				interpolate(isolevel, points[1], points[5], values[1], values[5]);
			if (edgeTable[cubeindex] & 1024)
				vertlist[10] =
				interpolate(isolevel, points[2], points[6], values[2], values[6]);
			if (edgeTable[cubeindex] & 2048)
				vertlist[11] =
				interpolate(isolevel, points[3], points[7], values[3], values[7]);

			// Create the triangle 
			Triangle triangle;
			for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
				triangle.vertices[0] = vertlist[triTable[cubeindex][i + 0]];
				triangle.vertices[1] = vertlist[triTable[cubeindex][i + 1]];
				triangle.vertices[2] = vertlist[triTable[cubeindex][i + 2]];

				triangles.push_back(triangle);
			}
			
		}

	};


};


int test()
{
	cv::viz::Viz3d window("VD-SPH");
	window.spinOnce(1, true);

	cv::Vec3f cam_pos(1.0f, 2.0f, 1.5f), cam_focal_point(0.0f, 0.0f, 0.25f), cam_y_dir(0.0f, 0.0f, -1.0f);
	cv::Affine3f cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
	window.setViewerPose(cam_pose);

	cv::Point3d bot_left = cv::Point3d(-0.5, -0.5, 0.0);
	cv::Point3d top_right = cv::Point3d(0.5, 0.5, 1.0);

	cv::viz::WCube domain(bot_left, top_right, true, cv::viz::Color::red());
	window.showWidget("domain", domain);

	cv::viz::WPlane plane(cv::Size2d(1.0, 1.0), cv::viz::Color::yellow());
	window.showWidget("plane", plane);

	window.spinOnce(0, true);

	half half_float_value = half(0.0f);

	//FILE* file = fopen("../../data/sph_000840.bin", "rb");
	FILE* file = fopen("../../data/sph_001200.bin", "rb"); 
	
	if (file)
	{
		std::vector<Particle> particles(N);
		size_t elements_read = fread(particles.data(), sizeof(Particle), N, file);
		assert(elements_read == N);
		fclose(file);

		
		//std::cout << particles[0].radius << std::endl;

		cv::Mat features(N, 3, CV_32FC1);
		for (int i = 0; i < N; ++i)
		{
			const cv::Point3f position{
				float(particles[i].position_x),
				float(particles[i].position_y),
				float(particles[i].position_z) };
			features.at<float>(i, 0) = position.x;
			features.at<float>(i, 1) = position.y;
			features.at<float>(i, 2) = position.z;
		}

		std::unique_ptr<SearchIndex> search_index = std::make_unique<SearchIndex>(features, cvflann::KDTreeSingleIndexParams(10, false));
		cvflann::SearchParams search_params;
		std::vector<Triangle> triangles;


		cv::Point3d current_step = bot_left;

		for (float x = bot_left.x; x < top_right.x; x += step)
		{
			std::cout << (x / (top_right.x - bot_left.x) * 100)+50 << std::endl;
			for (float y = bot_left.y; y < top_right.y; y += step)
			{
				for (float z = bot_left.z; z < top_right.z; z += step)
				{
					//std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;

					cv::Point3d current_step(x,y,z);
					current_step.x += (step / 2);
					current_step.y += (step / 2);
					current_step.z += (step / 2);

					VD::cube c(current_step);

					const float radius = step / 2.0f;

					for (int cube_vertex = 0; cube_vertex < 8; cube_vertex++)
					{

						std::vector<float> distances(N);
						std::vector<float> query{ c.points[cube_vertex].x, c.points[cube_vertex].y, c.points[cube_vertex].z };
						Neighbourhood indices(N);
						int n = search_index->radiusSearch(query, indices, distances, radius, search_params);

						float rho = 0;
						if (n > 0)
						{
							for (int idx = 0; idx < n; idx++)
							{
								int particle_index = indices[idx];
								float distance = distances[particle_index];
								Particle& p = particles[particle_index];

								// <0, 1> - real <0, 0.1>
								float smooth_factor = 0.05f;
								
								rho += p.get_compute_value() * cubic_spline(distance, smooth_factor);

							}

							c.values[cube_vertex] = rho;
						}
					}
					c.make_triangles(triangles);

				}
			}
		}

		std::vector<cv::Point3d> cloud;
		cloud.reserve(triangles.size() * 3);

		cv::Mat polygon = cv::Mat(1, int(triangles.size()) * 4, CV_32S);

		
		for (int i = 0, j = 0, k = 0; i < int(triangles.size()); ++i)
		{

			const Triangle& triangle = triangles[i];
			const cv::Point3f& v0 = cv::Point3f(triangle.vertices[0].x, triangle.vertices[0].y, triangle.vertices[0].z);
			const cv::Point3f& v1 = cv::Point3f(triangle.vertices[1].x, triangle.vertices[1].y, triangle.vertices[1].z);
			const cv::Point3f& v2 = cv::Point3f(triangle.vertices[2].x, triangle.vertices[2].y, triangle.vertices[2].z);

			cloud.push_back(cv::Point3f(v0.x, v0.y, v0.z));
			cloud.push_back(cv::Point3f(v1.x, v1.y, v1.z));
			cloud.push_back(cv::Point3f(v2.x, v2.y, v2.z));

			polygon.at<int>(0, j++) = 3; // vertices count
			polygon.at<int>(0, j++) = k++;
			polygon.at<int>(0, j++) = k++;
			polygon.at<int>(0, j++) = k++;
		}
		

		cv::viz::WMesh mesh(cloud, polygon);
		window.showWidget("Mesh", mesh);
		
		window.spin();
	}

	return 0;
}

int main()
{
	return test();	
}
