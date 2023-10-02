#pragma once
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <direct.h>
#include <fstream>  
#include <stdio.h>  
#include <string>  
#include <sstream>  
#include <vector> 
#include <thread>
#include <chrono>
#include <atomic>
#include <ctime> 
#include <cmath>
#include <mutex>
#include <map>
#include <set>
#include <io.h>
#include <memory>
#include <functional>
#include <cassert>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/io.h>
#include <pcl/surface/gp3.h>
#include <pcl/geometry/polygon_mesh.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <vtkVersion.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

using namespace std;

/** \brief Distortion model: defines how pixel coordinates should be mapped to sensor coordinates. */
typedef enum rs2_distortion
{
	RS2_DISTORTION_NONE, /**< Rectilinear images. No distortion compensation required. */
	RS2_DISTORTION_MODIFIED_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to radially distorted points */
	RS2_DISTORTION_INVERSE_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it */
	RS2_DISTORTION_FTHETA, /**< F-Theta fish-eye distortion model */
	RS2_DISTORTION_BROWN_CONRADY, /**< Unmodified Brown-Conrady distortion model */
	RS2_DISTORTION_KANNALA_BRANDT4, /**< Four parameter Kannala Brandt distortion model */
	RS2_DISTORTION_COUNT                   /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
} rs2_distortion;

/** \brief Video stream intrinsics. */
typedef struct rs2_intrinsics
{
	int           width;     /**< Width of the image in pixels */
	int           height;    /**< Height of the image in pixels */
	float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
	float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
	float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
	float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
	rs2_distortion model;    /**< Distortion model of the image */
	float         coeffs[5]; /**< Distortion coefficients */
} rs2_intrinsics;

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
static void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics* intrin, const float point[3])
{
	float x = point[0] / point[2], y = point[1] / point[2];

	if ((intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) ||
		(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY))
	{

		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		x *= f;
		y *= f;
		float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = dx;
		y = dy;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r * tan(intrin->coeffs[0] / 2.0f)));
		x *= rd / r;
		y *= rd / r;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float theta = atan(r);
		float theta2 = theta * theta;
		float series = 1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])));
		float rd = theta * series;
		x *= rd / r;
		y *= rd / r;
	}

	pixel[0] = x * intrin->fx + intrin->ppx;
	pixel[1] = y * intrin->fy + intrin->ppy;
}

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth)
{
	assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
	//assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

	float x = (pixel[0] - intrin->ppx) / intrin->fx;
	float y = (pixel[1] - intrin->ppy) / intrin->fy;
	if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
	{
		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = ux;
		y = uy;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}

		float theta = rd;
		float theta2 = rd * rd;
		for (int i = 0; i < 4; i++)
		{
			float f = theta * (1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])))) - rd;
			if (abs(f) < FLT_EPSILON)
			{
				break;
			}
			float df = 1 + theta2 * (3 * intrin->coeffs[0] + theta2 * (5 * intrin->coeffs[1] + theta2 * (7 * intrin->coeffs[2] + 9 * theta2 * intrin->coeffs[3])));
			theta -= f / df;
			theta2 = theta * theta;
		}
		float r = tan(theta);
		x *= r / rd;
		y *= r / rd;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}
		float r = (float)(tan(intrin->coeffs[0] * rd) / atan(2 * tan(intrin->coeffs[0] / 2.0f)));
		x *= r / rd;
		y *= r / rd;
	}

	point[0] = depth * x;
	point[1] = depth * y;
	point[2] = depth;
}

#define RandomIterative 0
#define RandomOneshot 1
#define EnsembleRGB 2
#define EnsembleRGBDensity 3
#define PVBCoverage 4

class Share_Data
{
public:
	//可变输入参数
	string model_path;
	string pcd_file_path;
	string ply_file_path;
	string yaml_file_path;
	string name_of_pcd;
	string nbv_net_path;
	string viewspace_path;
	string instant_ngp_path;
	string orginalviews_path;
	string shape_net;
	string pvb_path;

	int num_of_views;					//一次采样视点个数
	double cost_weight;
	rs2_intrinsics color_intrinsics;
	double depth_scale;
	double view_space_radius;
	int num_of_thread;

	//运行参数
	int process_cnt;					//过程编号
	atomic<double> pre_clock;			//系统时钟
	atomic<bool> over;					//过程是否结束
	bool show;
	int num_of_max_iteration;

	//点云数据
	atomic<int> vaild_clouds;
	vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds;							//点云组
	vector<vector<vector<double>> > pixel_nerf_rendering;
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pcd;
	pcl::PolygonMesh::Ptr mesh_ply;
	int mesh_data_offset;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ground_truth;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final;
	bool move_wait;
	map<string, double> mp_scale;

	//图片
	vector<cv::Mat> rgb_images;
	vector<cv::Mat> rgb_gt_images;
	vector<cv::Mat> depth_images;
	vector<cv::Mat> depth_gt_images;

	//pixel label (c_x,c_y,c_z,d_x,d_y,d_z,r,g,b) → images → views
	vector<vector<vector<float>>> view_image_labels;

	//八叉地图
	shared_ptr<octomap::ColorOcTree> octo_model;
	shared_ptr<octomap::ColorOcTree> ground_truth_model;
	shared_ptr<octomap::ColorOcTree> GT_sample;
	double octomap_resolution;
	double ground_truth_resolution;
	double map_size;
	double p_unknown_upper_bound; //! Upper bound for voxels to still be considered uncertain. Default: 0.97.
	double p_unknown_lower_bound; //! Lower bound for voxels to still be considered uncertain. Default: 0.12.
	
	//工作空间与视点空间
	atomic<bool> now_view_space_processed;
	atomic<bool> now_views_infromation_processed;
	atomic<bool> move_on;

	Eigen::Matrix4d now_camera_pose_world;
	Eigen::Vector3d object_center_world;		//物体中心
	double predicted_size;						//物体BBX半径

	int method_of_IG;

	double stop_thresh_map;
	double stop_thresh_view;

	double skip_coefficient;

	double sum_local_information;
	double sum_global_information;

	double sum_robot_cost;
	double camera_to_object_dis;
	bool robot_cost_negtive;

	int num_of_max_flow_node;
	double interesting_threshold;

	double see_threshold;
	double need_threshold;

	int init_voxels;     //点云voxel个数
	int full_voxels;     //点云voxel个数
	int voxels_in_BBX;   //地图voxel个数
	double init_entropy; //地图信息熵

	string pre_path;
	string gt_path;
	string save_path;
	string save_path_nbvnet;
	string save_path_pcnbv;

	vector<vector<double>> pt_sphere;
	double pt_norm;
	double min_z_table;

	vector<unsigned long long> view_cases;

	int ray_casting_aabb_scale;
	int num_of_novel_test_views;
	int num_of_neighbors_with_self;
	int num_of_choose;
	int num_of_random_test;
	int num_of_most_cover;
	int cost_on;
	double cost_rate;
	int visit_weight_type;
	double trunc_threshold;

	double approaching_threshold;
	int is_shape_net;
	int coverage_view_num_max;
	int coverage_view_num_add;
	int points_size_cloud;
	int n_steps;
	double object_pixel_rate;
	int id_of_batch;
	int ensemble_num;
	int evaluate;

	Share_Data(string _config_file_path, string test_name = "", int _num_of_views = -1, int _id_of_batch = -1, int test_method = -1)
	{
		process_cnt = -1;
		yaml_file_path = _config_file_path;
		//读取yaml文件
		cv::FileStorage fs;
		fs.open(yaml_file_path, cv::FileStorage::READ);
		fs["pre_path"] >> pre_path;
		fs["model_path"] >> model_path;
		fs["viewspace_path"] >> viewspace_path;
		fs["instant_ngp_path"] >> instant_ngp_path;
		fs["orginalviews_path"] >> orginalviews_path;
		fs["pvb_path"] >> pvb_path;
		fs["shape_net"] >> shape_net;
		fs["name_of_pcd"] >> name_of_pcd;
		fs["method_of_IG"] >> method_of_IG;
		fs["num_of_thread"] >> num_of_thread;
		fs["octomap_resolution"] >> octomap_resolution;
		fs["ground_truth_resolution"] >> ground_truth_resolution;
		fs["num_of_neighbors_with_self"] >> num_of_neighbors_with_self;
		fs["num_of_choose"] >> num_of_choose;
		fs["num_of_random_test"] >> num_of_random_test;
		fs["num_of_most_cover"] >> num_of_most_cover;
		fs["is_shape_net"] >> is_shape_net;
		fs["approaching_threshold"] >> approaching_threshold;
		fs["points_size_cloud"] >> points_size_cloud;
		fs["n_steps"] >> n_steps;
		fs["object_pixel_rate"] >> object_pixel_rate;
		fs["id_of_batch"] >> id_of_batch;
		fs["evaluate"] >> evaluate;
		fs["ensemble_num"] >> ensemble_num;
		fs["cost_on"] >> cost_on;
		fs["cost_rate"] >> cost_rate;
		fs["num_of_max_iteration"] >> num_of_max_iteration;
		fs["coverage_view_num_max"] >> coverage_view_num_max;
		fs["coverage_view_num_add"] >> coverage_view_num_add;
		fs["show"] >> show;
		fs["move_wait"] >> move_wait;
		fs["nbv_net_path"] >> nbv_net_path;
		fs["p_unknown_upper_bound"] >> p_unknown_upper_bound;
		fs["p_unknown_lower_bound"] >> p_unknown_lower_bound;
		fs["num_of_views"] >> num_of_views;
		fs["num_of_novel_test_views"] >> num_of_novel_test_views;
		fs["ray_casting_aabb_scale"] >> ray_casting_aabb_scale;
		fs["view_space_radius"] >> view_space_radius;
		fs["visit_weight_type"] >> visit_weight_type;
		fs["trunc_threshold"] >> trunc_threshold;
		fs["cost_weight"] >> cost_weight;
		fs["robot_cost_negtive"] >> robot_cost_negtive;
		fs["skip_coefficient"] >> skip_coefficient;
		fs["num_of_max_flow_node"] >> num_of_max_flow_node;
		fs["interesting_threshold"] >> interesting_threshold;
		fs["see_threshold"] >> see_threshold;
		fs["need_threshold"] >> need_threshold;
		fs["color_width"] >> color_intrinsics.width;
		fs["color_height"] >> color_intrinsics.height;
		fs["color_fx"] >> color_intrinsics.fx;
		fs["color_fy"] >> color_intrinsics.fy;
		fs["color_ppx"] >> color_intrinsics.ppx;
		fs["color_ppy"] >> color_intrinsics.ppy;
		fs["color_model"] >> color_intrinsics.model;
		fs["color_k1"] >> color_intrinsics.coeffs[0];
		fs["color_k2"] >> color_intrinsics.coeffs[1];
		fs["color_k3"] >> color_intrinsics.coeffs[2];
		fs["color_p1"] >> color_intrinsics.coeffs[3];
		fs["color_p2"] >> color_intrinsics.coeffs[4];
		fs["depth_scale"] >> depth_scale;
		fs.release();
		if (test_name != "") name_of_pcd = test_name;
		if (test_method != -1) method_of_IG = test_method;
		if (_num_of_views != -1) num_of_views = _num_of_views;
		if (_id_of_batch != -1) id_of_batch = _id_of_batch;
		if (!is_shape_net) {
			coverage_view_num_max = 90;
			coverage_view_num_add = 1;
		}
		//读取转换后模型的pcd文件
		pcd_file_path = model_path + "PCD/";
		//cloud_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
		cloud_pcd.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		
		//读取转换后模型的ply文件
		ply_file_path = model_path + "PLY/";
		mesh_ply.reset(new pcl::PolygonMesh);

		//调整大小保持水密表面
		mp_scale["Armadillo"] = 0.02;
		mp_scale["Asian_Dragon"] = 0.05;
		mp_scale["Dragon"] = 0.05;
		mp_scale["Stanford_Bunny"] = 0.04;
		mp_scale["Happy_Buddha"] = 0.07;
		mp_scale["Thai_Statue"] = 0.25;
		mp_scale["Lucy"] = 1.39;
		mp_scale["LM11"] = 0.03;
		mp_scale["LM12"] = 0.04;
		mp_scale["obj_000001"] = 0.02;
		mp_scale["obj_000002"] = 0.06;
		mp_scale["obj_000004"] = 0.02;
		mp_scale["obj_000005"] = 0.02;
		mp_scale["obj_000007"] = 0.05;
		mp_scale["obj_000008"] = 0.03;
		mp_scale["obj_000009"] = 0.03;
		mp_scale["obj_000010"] = 0.03;
		mp_scale["obj_000011"] = 0.06;
		mp_scale["obj_000012"] = 0.02;
		mp_scale["obj_000018"] = 0.02;
		mp_scale["obj_000020"] = 0.08;
		mp_scale["obj_000021"] = 0.03;
		mp_scale["obj_000022"] = 0.02;
		mp_scale["obj_000023"] = 0.04;
		mp_scale["obj_000024"] = 0.05;
		mp_scale["obj_000025"] = 0.05;
		mp_scale["obj_000026"] = 0.01;
		mp_scale["obj_000027"] = 0.09;
		mp_scale["obj_000028"] = 0.17;
		mp_scale["obj_000029"] = 0.02;
		mp_scale["obj_000030"] = 0.18;

		//octo_model = new octomap::ColorOcTree(octomap_resolution);
		//octo_model->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//octo_model->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//octo_model->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//octo_model->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192
		//octo_model->setOccupancyThres(0.5);	//设置节点占用阈值，初始0.5
		ground_truth_model = make_shared<octomap::ColorOcTree>(ground_truth_resolution);
		//ground_truth_model->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//ground_truth_model->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//ground_truth_model->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//ground_truth_model->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192
		//GT_sample = new octomap::ColorOcTree(octomap_resolution);
		//GT_sample->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//GT_sample->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//GT_sample->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//GT_sample->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192
		/*cloud_model = new octomap::ColorOcTree(ground_truth_resolution);
		//cloud_model->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//cloud_model->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//cloud_model->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//cloud_model->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192
		//cloud_model->setOccupancyThres(0.5);	//设置节点占用阈值，初始0.5*/
		if (num_of_max_flow_node == -1) num_of_max_flow_node = num_of_views;
		now_camera_pose_world = Eigen::Matrix4d::Identity(4, 4);
		over = false;
		pre_clock = clock();
		vaild_clouds = 0;
		cloud_final.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_ground_truth.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		//path
		gt_path = pre_path + "Coverage_images/";
		save_path = pre_path + "Compare/" ;
		
		if (is_shape_net) {
			gt_path += "ShapeNet";
			save_path += "ShapeNet";

			if (id_of_batch >= 0) {
				gt_path += "_" + to_string(id_of_batch);
				save_path += "_" + to_string(id_of_batch);
			}

			gt_path += "/";
			save_path += "/";
		}

		gt_path += name_of_pcd;
		save_path += name_of_pcd;

		if (test_method != -1) {
			save_path += "_m" + to_string(method_of_IG);
		}

		if (method_of_IG == 2) {
			ensemble_num = 2; //Use the value in the paper
		}
		else if (method_of_IG == 3) {
			ensemble_num = 5; //Use the value in the paper
		}

		cout << "gt_path is: " << gt_path << endl;
		cout << "save_path is: " << save_path << endl;
		srand(clock());
		//read viewspace
		//ifstream fin_sphere("../view_space_" + to_string(num_of_views) + ".txt");
		ifstream fin_sphere(viewspace_path + to_string(num_of_views) + ".txt");
		pt_sphere.resize(num_of_views);
		for (int i = 0; i < num_of_views; i++) {
			pt_sphere[i].resize(3);
			for (int j = 0; j < 3; j++) {
				fin_sphere >> pt_sphere[i][j];
				//cout << pt_sphere[i][j] << " ??? " << endl;
			}
		}
		cout<< "view space size is: " << pt_sphere.size() << endl;
		Eigen::Vector3d pt0(pt_sphere[0][0], pt_sphere[0][1], pt_sphere[0][2]);
		pt_norm = pt0.norm();
		//read view cases
		ifstream fin_view_cases;
		fin_view_cases.open(pre_path + "/view_cases.txt");
		unsigned long long cas;
		while (fin_view_cases >> cas) {
			view_cases.push_back(cas);
		}
		cout << "test view case num is: " << view_cases.size() << endl;
	}

	~Share_Data() {
		//释放内存
		pixel_nerf_rendering.clear();
		pixel_nerf_rendering.shrink_to_fit();
		rgb_images.clear();
		rgb_images.shrink_to_fit();
		rgb_gt_images.clear();
		rgb_gt_images.shrink_to_fit();
		depth_images.clear();
		depth_images.shrink_to_fit();
		depth_gt_images.clear();
		depth_gt_images.shrink_to_fit();
		view_image_labels.clear();
		view_image_labels.shrink_to_fit();
		pt_sphere.clear();
		pt_sphere.shrink_to_fit();
		view_cases.clear();
		view_cases.shrink_to_fit();
		octo_model.reset();
		ground_truth_model.reset();
		GT_sample.reset();
		cloud_pcd->points.clear();
		cloud_pcd->points.shrink_to_fit();
		cloud_pcd.reset();
		mesh_ply->cloud.data.clear();
		mesh_ply->cloud.data.shrink_to_fit();
		mesh_ply.reset();
		cloud_ground_truth->points.clear();
		cloud_ground_truth->points.shrink_to_fit();
		cloud_ground_truth.reset();
		cloud_final->points.clear();
		cloud_final->points.shrink_to_fit();
		cloud_final.reset();
		for (int i = 0; i < clouds.size(); i++) {
			clouds[i]->points.clear();
			clouds[i]->points.shrink_to_fit();
			clouds[i].reset();
		}
		//for (int i = 0; i < clouds.size(); i++) {
		//	cout << "Share_Data: clouds[" << i << "] use_count is " << clouds[i].use_count() << endl;
		//}
		clouds.clear();
		clouds.shrink_to_fit();
		//cout << "Share_Data: octo_model use_count is " << octo_model.use_count() << endl;
		//cout << "Share_Data: ground_truth_model use_count is " << ground_truth_model.use_count() << endl;
		//cout << "Share_Data: GT_sample use_count is " << GT_sample.use_count() << endl;
		//cout << "Share_Data: cloud_pcd use_count is " << cloud_pcd.use_count() << endl;
		//cout << "Share_Data: mesh_ply use_count is " << mesh_ply.use_count() << endl;
		//cout << "Share_Data: cloud_ground_truth use_count is " << cloud_ground_truth.use_count() << endl;
		//cout << "Share_Data: cloud_final use_count is " << cloud_final.use_count() << endl;
	}

	Eigen::Matrix4d get_toward_pose(int toward_state)
	{
		Eigen::Matrix4d pose(Eigen::Matrix4d::Identity(4, 4));
		switch (toward_state) {
			case 0://z<->z
				return pose;
			case 1://z<->-z
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = 0; pose(2, 2) = -1; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 2://z<->x
				pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = 1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 3://z<->-x
				pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = -1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 4://z<->y
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = 1; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 5://z<->-y
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = -1; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
		}
		return pose;
	}

	double out_clock()
	{   //返回用时，并更新时钟
		double now_clock = clock();
		double time_len = now_clock - pre_clock;
		pre_clock = now_clock;
		return time_len;
	}

	void access_directory(string cd)
	{   //检测多级目录的文件夹是否存在，不存在就创建
		string temp;
		for (int i = 0; i < cd.length(); i++)
			if (cd[i] == '/') {
				if (access(temp.c_str(), 0) != 0) mkdir(temp.c_str());
				temp += cd[i];
			}
			else temp += cd[i];
		if (access(temp.c_str(), 0) != 0) mkdir(temp.c_str());
	}

	void save_posetrans_to_disk(Eigen::Matrix4d& T, string cd, string name, int frames_cnt)
	{   //存放旋转矩阵数据至磁盘
		std::stringstream pose_stream, path_stream;
		std::string pose_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		pose_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << frames_cnt << ".txt";
		pose_stream >> pose_file;
		ofstream fout(pose_file);
		fout << T;
	}

	void save_octomap_log_to_disk(int voxels, double entropy, string cd, string name,int iterations)
	{
		std::stringstream log_stream, path_stream;
		std::string log_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		log_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << iterations << ".txt";
		log_stream >> log_file;
		ofstream fout(log_file);
		fout << voxels << " " << entropy << endl;
	}

	void save_cloud_to_disk(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string cd, string name)
	{   //存放点云数据至磁盘，速度很慢，少用
		std::stringstream cloud_stream, path_stream;
		std::string cloud_file, path;
		path_stream << save_path << cd;
		path_stream >> path;
		access_directory(path);
		cloud_stream << save_path << cd << "/" << name << ".pcd";
		cloud_stream >> cloud_file;
		pcl::io::savePCDFile<pcl::PointXYZRGB>(cloud_file, *cloud);
	}

	void save_cloud_to_disk(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string cd, string name, int frames_cnt)
	{   //存放点云数据至磁盘，速度很慢，少用
		std::stringstream cloud_stream, path_stream;
		std::string cloud_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		cloud_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << frames_cnt << ".pcd";
		cloud_stream >> cloud_file;
		pcl::io::savePCDFile<pcl::PointXYZRGB>(cloud_file, *cloud);
	}

	void save_octomap_to_disk(octomap::ColorOcTree* octo_model, string cd, string name)
	{   //存放点云数据至磁盘，速度很慢，少用
		std::stringstream octomap_stream, path_stream;
		std::string octomap_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		octomap_stream << "../data" << "_" << process_cnt << cd << "/" << name << ".ot";
		octomap_stream >> octomap_file;
		octo_model->write(octomap_file);
	}

};

inline double pow2(double x) {
	return x * x;
}

inline octomap::point3d project_pixel_to_ray_end(int x, int y, rs2_intrinsics& color_intrinsics, Eigen::Matrix4d& now_camera_pose_world, float max_range) {
	float pixel[2] = { float(x),float(y) };
	float point[3];
	rs2_deproject_pixel_to_point(point, &color_intrinsics, pixel, max_range);
	Eigen::Vector4d point_world(point[0], point[1], point[2], 1);
	point_world = now_camera_pose_world * point_world;
	return octomap::point3d(point_world(0), point_world(1), point_world(2));
}

vector<string> getFilesList(string dir)
{
	vector<string> allPath;
	// 在目录后面加上"\\*.*"进行第一次搜索
	string dir2 = dir + "/*.*";

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dir2.c_str(), &findData);
	if (handle == -1) {// 检查是否成功
		cout << "can not found the file ... " << endl;
		return allPath;
	}
	do
	{
		if (findData.attrib & _A_SUBDIR) //是否含有子目录
		{
			//若该子目录为"."或".."，则进行下一次循环，否则输出子目录名，并进入下一次搜索
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;

			// 在目录后面加上"\\"和搜索到的目录名进行下一次搜索
			string dirNew = dir + "/" + findData.name;
			vector<string> tempPath = getFilesList(dirNew);
			allPath.insert(allPath.end(), tempPath.begin(), tempPath.end());
		}
		else //不是子目录，即是文件，则输出文件名和文件的大小
		{
			string filePath = dir + "/" + findData.name;
			string fileName = findData.name;
			if (fileName == "model_normalized.json") {
				allPath.push_back(dir.substr(1));
				//cout << dir.substr(42) << endl;
				//cout << filePath << "\t" << findData.size << " bytes.\n";
			}
		}
	} while (_findnext(handle, &findData) == 0);
	_findclose(handle);    // 关闭搜索句柄
	return allPath;
}

//转换白背景为透明
void convertToAlpha(cv::Mat& src, cv::Mat& dst)
{
	cv::cvtColor(src, dst, cv::COLOR_BGR2BGRA);
	for (int y = 0; y < dst.rows; ++y)
	{
		for (int x = 0; x < dst.cols; ++x)
		{
			cv::Vec4b& pixel = dst.at<cv::Vec4b>(y, x);
			if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255){
				pixel[3] = 0;
			}
		}
	}
}

//色彩化深度图
void colorize_depth(cv::Mat& src, cv::Mat& dst)
{
	dst=src.clone();
	dst.convertTo(dst, CV_32F);
	double min, max;
	cv::minMaxIdx(dst, &min, &max);
	convertScaleAbs(dst, dst, 255 / max);
	applyColorMap(dst, dst, cv::COLORMAP_JET);
	for (int y = 0; y < dst.rows; ++y)
	{
		for (int x = 0; x < dst.cols; ++x)
		{
			unsigned short depth = src.at<unsigned short>(y, x);
			if (depth == 0) {
				dst.at<cv::Vec3b>(y, x)[0] = 255;
				dst.at<cv::Vec3b>(y, x)[1] = 255;
				dst.at<cv::Vec3b>(y, x)[2] = 255;
			}
		}
	}
}

double ColorfulNess(cv::Mat& frame)
{
	// split image to 3 channels (B,G,R)
	cv::Mat channelsBGR[3];
	cv::split(frame, channelsBGR);

	// rg = R - G
	// yb = 0.5*(R + G) - B
	cv::Mat rg, yb;
	cv::absdiff(channelsBGR[2], channelsBGR[1], rg);
	cv::absdiff(0.5 * (channelsBGR[2] + channelsBGR[1]), channelsBGR[0], yb);

	// calculate the mean and std for rg and yb
	cv::Mat rgMean, rgStd; // 1*1矩阵
	cv::meanStdDev(rg, rgMean, rgStd);
	cv::Mat ybMean, ybStd; // 1*1矩阵
	cv::meanStdDev(yb, ybMean, ybStd);

	// calculate the mean and std for rgyb
	double stdRoot, meanRoot;
	stdRoot = sqrt(pow(rgStd.at<double>(0, 0), 2)
		+ pow(ybStd.at<double>(0, 0), 2));
	meanRoot = sqrt(pow(rgMean.at<double>(0, 0), 2)
		+ pow(ybMean.at<double>(0, 0), 2));

	// return colorfulNess
	return stdRoot + (0.3 * meanRoot);
}

/*

string name_x;
cin >> name_x;
double object_rate = 0.0;
for (int view_id = 0; view_id < 5; view_id++) {
	cv::Mat img_size_test = cv::imread("D:/Data/NeRF_coverage/Coverage_images/ShapeNet/" + name_x +"/5/rgbaClip_" + to_string(view_id) + ".png");
	int count_object = 0;
	for (int i = 0; i < img_size_test.rows; i++) {
		for (int j = 0; j < img_size_test.cols; j++) {
			if (img_size_test.at<cv::Vec3b>(i, j)[0] != 255 || img_size_test.at<cv::Vec3b>(i, j)[1] != 255 || img_size_test.at<cv::Vec3b>(i, j)[2] != 255) {
				count_object++;
			}
		}
	}
	object_rate += (double)count_object / (img_size_test.rows * img_size_test.cols);
}
object_rate /= 5;
cout << "object rate is " << object_rate << endl;

*/