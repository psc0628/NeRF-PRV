#include <windows.h>
#include <iostream>
#include <cstdio>
#include <thread>
#include <atomic>
#include <chrono>
typedef unsigned unsigned long long pop_t;

using namespace std;

#include "Share_Data.hpp"
#include "View_Space.hpp"
#include <gurobi_c++.h>
#include "json/json.h"

//Virtual_Perception_3D.hpp
class Perception_3D {
public:
	shared_ptr<Share_Data> share_data;
	shared_ptr<octomap::ColorOcTree> ground_truth_model;
	int full_voxels;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	Eigen::Matrix4d view_pose_world;
	octomap::point3d origin;
	vector<octomap::point3d> end;
	pcl::visualization::PCLVisualizer::Ptr viewer;

	Perception_3D(shared_ptr<Share_Data>& _share_data) {
		share_data = _share_data;
		ground_truth_model = share_data->ground_truth_model;
		full_voxels = share_data->full_voxels;
		view_pose_world = Eigen::Matrix4d::Identity();
		cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		viewer.reset(new pcl::visualization::PCLVisualizer("Render"));
		viewer->setBackgroundColor(255, 255, 255);
		viewer->initCameraParameters();
		if (share_data->is_shape_net) {
			viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_gt");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, share_data->points_size_cloud, "cloud_gt");
		}
		else {
			viewer->addPolygonMesh(*share_data->mesh_ply, "mesh_ply");
		}
	}

	~Perception_3D() {
		if (share_data->is_shape_net) {
			viewer->removePointCloud("cloud_gt");
		}
		else {
			viewer->removePolygonMesh("mesh_ply");
		}
		viewer->close();
		viewer.reset();
		end.clear();
		end.shrink_to_fit();
		share_data.reset();
		ground_truth_model.reset();
		cloud->points.clear();
		cloud->points.shrink_to_fit();
		cloud.reset();
		//cout << "Perception_3D: share_data use_count is " << share_data.use_count() << endl;
		//cout << "Perception_3D: ground_truth_model use_count is " << ground_truth_model.use_count() << endl;
		//cout << "Perception_3D: cloud use_count is " << cloud.use_count() << endl;
		//cout << "Perception_3D: viewer use_count is " << viewer.use_count() << endl;
	}

	bool render(View& now_best_view,int id, string path = "") {
		//获取视点位姿
		Eigen::Matrix4d view_pose_world;
		now_best_view.get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		view_pose_world = (share_data->now_camera_pose_world * now_best_view.pose.inverse()).eval();
		//渲染
		Eigen::Matrix3f intrinsics;
		intrinsics << share_data->color_intrinsics.fx, 0, share_data->color_intrinsics.ppx,
			0, share_data->color_intrinsics.fy, share_data->color_intrinsics.ppy,
			0, 0, 1;
		Eigen::Matrix4f extrinsics = view_pose_world.cast<float>();
		viewer->setCameraParameters(intrinsics, extrinsics);
		pcl::visualization::Camera cam;
		viewer->getCameraParameters(cam);
		cam.window_size[0] = share_data->color_intrinsics.width;
		cam.window_size[1] = share_data->color_intrinsics.height;
		viewer->setCameraParameters(cam);
		viewer->spinOnce(100);
		share_data->access_directory(share_data->gt_path + path);
		viewer->saveScreenshot(share_data->gt_path + path + "/rgb_" + to_string(id) + ".png");

		//viewer->addCoordinateSystem(1.0);
		//while (!viewer->wasStopped()) {
		//	viewer->spinOnce(100);
		//	this_thread::sleep_for(chrono::milliseconds(100));
		//}

		return true;
	}

	bool precept(View& now_best_view) { 
		double now_time = clock();
		//创建当前成像点云
		cloud->points.clear();
		cloud->points.shrink_to_fit();
		cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud->is_dense = false;
		cloud->points.resize(full_voxels);
		
		//获取视点位姿
		now_best_view.get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		view_pose_world = (share_data->now_camera_pose_world * now_best_view.pose.inverse()).eval();
		//检查视点的key
		octomap::OcTreeKey key_origin;
		bool key_origin_have = ground_truth_model->coordToKeyChecked(now_best_view.init_pos(0), now_best_view.init_pos(1), now_best_view.init_pos(2), key_origin);
		if (key_origin_have) {
			origin = ground_truth_model->keyToCoord(key_origin);
			//遍历每个体素
			end.resize(full_voxels);
			octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs();
			for (int i = 0; i < full_voxels; i++) {
				end[i] = it.getCoordinate();
				it++;
			}
			//ground_truth_model->write(share_data->save_path + "/test_camrea.ot");
			//多线程处理
			vector<thread> precept_process;
			for (int i = 0; i < full_voxels; i+= share_data->num_of_thread) {
				for (int j = 0; j < share_data->num_of_thread && i + j < full_voxels; j++)
					precept_process.push_back(thread(bind(&Perception_3D::precept_thread_process, this, i + j)));
				for (int j = 0; j < share_data->num_of_thread && i + j < full_voxels; j++)
					precept_process[i + j].join();
			}

			//释放内存
			precept_process.clear();
			precept_process.shrink_to_fit();
			end.clear();
			end.shrink_to_fit();
		}
		else {
			cout << "View out of map.check." << endl;
		}
		/*
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
		temp->is_dense = false;
		temp->points.resize(full_voxels);
		auto ptr = temp->points.begin();
		int vaild_point = 0;
		auto p = cloud->points.begin();
		for (int i = 0; i < cloud->points.size(); i++, p++)
		{
			if ((*p).x == 0 && (*p).y == 0 && (*p).z == 0) continue;
			(*ptr).x = (*p).x;
			(*ptr).y = (*p).y;
			(*ptr).z = (*p).z;
			(*ptr).b = (*p).b;
			(*ptr).g = (*p).g;
			(*ptr).r = (*p).r;
			vaild_point++;
			ptr++;
		}
		temp->width = vaild_point;
		temp->height = 1;
		temp->points.resize(vaild_point);
		*/
		//记录当前采集点云
		share_data->vaild_clouds++;
		//share_data->clouds.push_back(temp);
		//旋转至世界坐标系
		//*share_data->cloud_final += *temp;
		cout << "virtual cloud get with executed time " << clock() - now_time << " ms." << endl;
		if (share_data->show) { //显示成像点云
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
			viewer1->setBackgroundColor(255, 255, 255);
			//viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			Eigen::Matrix3f intrinsics;
			intrinsics << share_data->color_intrinsics.fx, 0, share_data->color_intrinsics.ppx,
				0, share_data->color_intrinsics.fy, share_data->color_intrinsics.ppy,
				0, 0, 1;
			Eigen::Matrix4f extrinsics = view_pose_world.cast<float>();
			viewer1->setCameraParameters(intrinsics, extrinsics);
			pcl::visualization::Camera cam;
			viewer1->getCameraParameters(cam);
			cam.window_size[0] = share_data->color_intrinsics.width;
			cam.window_size[1] = share_data->color_intrinsics.height;
			viewer1->setCameraParameters(cam);
			viewer1->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
			Eigen::Vector4d X(0.05, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.05, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.05, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			viewer1->spinOnce(100);
			viewer1->saveScreenshot(share_data->save_path + "/cloud.png");
			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
			viewer1->removeAllCoordinateSystems();
			viewer1->removeAllPointClouds();
			viewer1->removeAllShapes();
			viewer1->close();
			viewer1.reset();

			//显示了投影图像是180转置
			cv::Mat color = cv::Mat(share_data->color_intrinsics.height, share_data->color_intrinsics.width, CV_8UC3);
			for (int y = 0; y < color.rows; ++y) {
				for (int x = 0; x < color.cols; ++x) {
					cv::Vec3b& pixel = color.at<cv::Vec3b>(y, x);
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
				}
			}
			for (int i = 0; i < cloud->points.size(); i++) {
				Eigen::Vector4d end_3d(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
				auto vertex = view_pose_world.inverse() * end_3d;
				float point_3d[3] = { vertex(0), vertex(1),vertex(2) };
				float pixel[2];
				rs2_project_point_to_pixel(pixel, &share_data->color_intrinsics, point_3d);
				color.at<cv::Vec3b>(pixel[1], pixel[0])[0] = 0;
				color.at<cv::Vec3b>(pixel[1], pixel[0])[1] = 0;
				color.at<cv::Vec3b>(pixel[1], pixel[0])[2] = 0;
			}
			cv::imshow("color", color);
			cv::waitKey(0);
		}

		return true;
	}

	void precept_thread_process(int i) {
		//num++;
		pcl::PointXYZRGB point;
		point.x = 0; point.y = 0; point.z = 0;
		//投影检测是否在成像范围内
		Eigen::Vector4d end_3d(end[i].x(), end[i].y(), end[i].z(), 1);
		Eigen::Vector4d vertex = view_pose_world.inverse() * end_3d;
		float point_3d[3] = { vertex(0), vertex(1),vertex(2) };
		float pixel[2];
		rs2_project_point_to_pixel(pixel, &share_data->color_intrinsics, point_3d);
		if (pixel[0] < 0 || pixel[0]>share_data->color_intrinsics.width || pixel[1] < 0 || pixel[1]>share_data->color_intrinsics.height) {
			cloud->points[i] = point;
			return;
		}
		//反向投影找到终点
		octomap::point3d end = project_pixel_to_ray_end(pixel[0], pixel[1], share_data->color_intrinsics, view_pose_world, 1.0);
		octomap::OcTreeKey key_end;
		octomap::point3d direction = end - origin;
		octomap::point3d end_point;
		//越过未知区域，找到终点
		bool found_end_point = ground_truth_model->castRay(origin, direction, end_point, true, 1.0);
		if (!found_end_point) {//未找到终点，无观测数据
			cloud->points[i] = point;
			return;
		}
		if (end_point == origin) {
			cout << "view in the object. check!" << endl;
			cloud->points[i] = point;
			return;
		}
		//检查一下末端是否在地图限制范围内
		bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
		if (key_end_have) {
			octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
			if (node != NULL) {
				octomap::ColorOcTreeNode::Color color = node->getColor();
				point.x = end_point.x();
				point.y = end_point.y();
				point.z = end_point.z();
				point.b = color.b;
				point.g = color.g;
				point.r = color.r;
				node = NULL;
			}
		}
		cloud->points[i] = point;
	}

};

//Global_Path_Planner.hpp

/* Solve a traveling salesman problem on a randomly generated set of
	points using lazy constraints.   The base MIP model only includes
	'degree-2' constraints, requiring each node to have exactly
	two incident edges.  Solutions to this model may contain subtours -
	tours that don't visit every node.  The lazy constraint callback
	adds new constraints to cut them off. */

// Given an integer-feasible solution 'sol', find the smallest
// sub-tour.  Result is returned in 'tour', and length is
// returned in 'tourlenP'.
void findsubtour(int n, double** sol, int* tourlenP, int* tour) {
	{
		bool* seen = new bool[n];
		int bestind, bestlen;
		int i, node, len, start;

		for (i = 0; i < n; i++)
			seen[i] = false;

		start = 0;
		bestlen = n + 1;
		bestind = -1;
		node = 0;
		while (start < n) {
			for (node = 0; node < n; node++)
				if (!seen[node])
					break;
			if (node == n)
				break;
			for (len = 0; len < n; len++) {
				tour[start + len] = node;
				seen[node] = true;
				for (i = 0; i < n; i++) {
					if (sol[node][i] > 0.5 && !seen[i]) {
						node = i;
						break;
					}
				}
				if (i == n) {
					len++;
					if (len < bestlen) {
						bestlen = len;
						bestind = start;
					}
					start += len;
					break;
				}
			}
		}

		for (i = 0; i < bestlen; i++)
			tour[i] = tour[bestind + i];
		*tourlenP = bestlen;

		delete[] seen;
	}
}

// Subtour elimination callback.  Whenever a feasible solution is found,
// find the smallest subtour, and add a subtour elimination constraint
// if the tour doesn't visit every node.
class subtourelim : public GRBCallback
{
public:
	GRBVar** vars;
	int n;
	subtourelim(GRBVar** xvars, int xn) {
		vars = xvars;
		n = xn;
	}
protected:
	void callback() {
		try {
			if (where == GRB_CB_MIPSOL) {
				// Found an integer feasible solution - does it visit every node?
				double** x = new double* [n];
				int* tour = new int[n];
				int i, j, len;
				for (i = 0; i < n; i++)
					x[i] = getSolution(vars[i], n);

				findsubtour(n, x, &len, tour);

				if (len < n) {
					// Add subtour elimination constraint
					GRBLinExpr expr = 0;
					for (i = 0; i < len; i++)
						for (j = i + 1; j < len; j++)
							expr += vars[tour[i]][tour[j]];
					addLazy(expr <= len - 1);
				}

				for (i = 0; i < n; i++)
					delete[] x[i];
				delete[] x;
				delete[] tour;
			}
		}
		catch (GRBException e) {
			cout << "Error number: " << e.getErrorCode() << endl;
			cout << e.getMessage() << endl;
		}
		catch (...) {
			cout << "Error during callback" << endl;
		}
	}
};

class Global_Path_Planner {
public:
	shared_ptr<Share_Data> share_data;
	int now_view_id;
	int end_view_id;
	bool solved;
	int n;
	map<int, int>* view_id_in;
	map<int, int>* view_id_out;
	vector<vector<double>> graph;
	double total_shortest;
	vector<int> global_path;
	GRBEnv* env = NULL;
	GRBVar** vars = NULL;
	GRBModel* model = NULL;
	subtourelim* cb = NULL;
	
	Global_Path_Planner(shared_ptr<Share_Data> _share_data, vector<View>& views, vector<int>& view_set_label, int _now_view_id, int _end_view_id = -1) {
		share_data = _share_data;
		now_view_id = _now_view_id;
		end_view_id = _end_view_id;
		solved = false;
		total_shortest = -1;
		//构造下标映射
		view_id_in = new map<int, int>();
		view_id_out = new map<int, int>();
		for (int i = 0; i < view_set_label.size(); i++) {
			(*view_id_in)[view_set_label[i]] = i;
			(*view_id_out)[i] = view_set_label[i];
		}
		(*view_id_in)[views.size()] = view_set_label.size(); //注意复制节点应该是和视点空间个数相关，映射到所需视点个数
		(*view_id_out)[view_set_label.size()] = views.size(); 
		//节点数为原始个数+起点的复制节点
		n = view_set_label.size() + 1;
		//local path 完全无向图
		graph.resize(n);
		for (int i = 0; i < n; i++)
			graph[i].resize(n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				if (i == n - 1 || j == n - 1) {
					//如果是起点的复制节点，距离为0
					graph[i][j] = 0.0;
				}
				else {
					//交换id
					int u = (*view_id_out)[i];
					int v = (*view_id_out)[j];
					//求两点路径
					pair<int, double> local_path = get_local_path(views[u].init_pos.eval(), views[v].init_pos.eval(), (share_data->object_center_world + Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), share_data->predicted_size);
					if (local_path.first < 0) {
						cout << "local path not found." << endl;
						graph[i][j] = 1e10;
					}
					else graph[i][j] = local_path.second;
				}
				//cout << "graph[" << i << "][" << j << "] = " << graph[i][j] << endl;
			}
		//创建Gurobi的TSP优化器
		vars = new GRBVar * [n];
		for (int i = 0; i < n; i++)
			vars[i] = new GRBVar[n];
		env = new GRBEnv();
		model = new GRBModel(*env);
		//cout << "Gurobi model created." << endl;
		// Must set LazyConstraints parameter when using lazy constraints
		model->set(GRB_IntParam_LazyConstraints, 1);
		//cout << "Gurobi set LazyConstraints." << endl;
		// Create binary decision variables
		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {
				vars[i][j] = model->addVar(0.0, 1.0, graph[i][j], GRB_BINARY, "x_" + to_string(i) + "_" + to_string(j));
				vars[j][i] = vars[i][j];
			}
		}
		//cout << "Gurobi addVar complete." << endl;
		// Degree-2 constraints
		for (int i = 0; i < n; i++) {
			GRBLinExpr expr = 0;
			for (int j = 0; j < n; j++)
				expr += vars[i][j];
			model->addConstr(expr == 2, "deg2_" + to_string(i));
		}
		//cout << "Gurobi add Degree-2 Constr complete." << endl;
		// Forbid edge from node back to itself
		for (int i = 0; i < n; i++)
			vars[i][i].set(GRB_DoubleAttr_UB, 0);
		//cout << "Gurobi set Forbid edge complete." << endl;
		// Make copy node linked to starting node
		vars[n - 1][(*view_id_in)[now_view_id]].set(GRB_DoubleAttr_LB, 1);
		// 默认不设置终点，多解只返回一个
		if(end_view_id != -1) vars[(*view_id_in)[end_view_id]][n - 1].set(GRB_DoubleAttr_LB, 1);
		//cout << "Gurobi set Make copy node complete." << endl;
		// Set callback function
		cb = new subtourelim(vars, n);
		model->setCallback(cb);
		//cout << "Gurobi set callback function complete." << endl;
		cout << "Global_Path_Planner inited." << endl;
	}

	~Global_Path_Planner() {
		graph.clear();
		graph.shrink_to_fit();
		global_path.clear();
		global_path.shrink_to_fit();
		for (int i = 0; i < n; i++)
			delete[] vars[i];
		delete[] vars;
		delete env;
		delete model;
		delete cb;
	}

	double solve() {
		double now_time = clock();
		// Optimize model
		model->optimize();
		// Extract solution
		if (model->get(GRB_IntAttr_SolCount) > 0) {
			solved = true;
			total_shortest = 0.0;
			double** sol = new double* [n];
			for (int i = 0; i < n; i++)
				sol[i] = model->get(GRB_DoubleAttr_X, vars[i], n);

			int* tour = new int[n];
			int len;

			findsubtour(n, sol, &len, tour);
			assert(len == n);

			//cout << "Tour: ";
			for (int i = 0; i < len; i++) {
				global_path.push_back(tour[i]);
				if (i != len - 1) {
					total_shortest += graph[tour[i]][tour[i + 1]];
				}
				else {
					total_shortest += graph[tour[i]][tour[0]];
				}
				//cout << tour[i] << " ";
			}
			//cout << endl;

			for (int i = 0; i < n; i++)
				delete[] sol[i];
			delete[] sol;
			delete[] tour;
		}
		else {
			cout << "No solution found" << endl;
		}
		double cost_time = clock() - now_time;
		cout << "Global Path length " << total_shortest << " getted with executed time " << cost_time << " ms." << endl;
		return total_shortest;
	}

	vector<int> get_path_id_set() {
		if (!solved) cout << "call solve() first" << endl;
		cout << "Node ids on global_path form start to end are: ";
		vector<int> ans = global_path;
		//调准顺序把复制的起点置于末尾
		int copy_node_id = -1;
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == n - 1) {
				copy_node_id = i;
				break;
			}
		}
		if (copy_node_id == -1) {
			cout << "copy_node_id == -1" << endl;
		}
		for (int i = 0; i < copy_node_id; i++) {
			ans.push_back(ans[0]);
			ans.erase(ans.begin());
		}
		//删除复制的起点
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == n - 1) {
				ans.erase(ans.begin() + i);
				break;
			}
		}
		//如果起点是第一个就不动，是最后一个就反转
		if (ans[0] != (*view_id_in)[now_view_id]) {
			reverse(ans.begin(), ans.end());
		}
		for (int i = 0; i < ans.size(); i++) {
			ans[i] = (*view_id_out)[ans[i]];
			cout << ans[i] << " ";
		}
		cout << endl;
		//删除出发点
		//ans.erase(ans.begin());
		return ans;
	}
};

//NBV_Net_Labeler.hpp
class NBV_Net_Labeler 
{
public:
	shared_ptr<Share_Data> share_data;
	shared_ptr<View_Space> view_space;
	shared_ptr<Perception_3D> percept;
	int toward_state;
	int rotate_state;
	bool object_is_ok_size;
	vector<View> init_views;

	double check_size(double predicted_size, Eigen::Vector3d object_center_world, vector<Eigen::Vector3d>& points) {
		int vaild_points = 0;
		for (auto& ptr : points) {
			if (ptr(0) < object_center_world(0) - predicted_size || ptr(0) > object_center_world(0) + predicted_size) continue;
			if (ptr(1) < object_center_world(1) - predicted_size || ptr(1) > object_center_world(1) + predicted_size) continue;
			if (ptr(2) < object_center_world(2) - predicted_size || ptr(2) > object_center_world(2) + predicted_size) continue;
			vaild_points++;
		}
		return (double)vaild_points / (double)points.size();
	}

	~NBV_Net_Labeler() {
		share_data.reset();
		view_space.reset();
		percept.reset();
		init_views.clear();
		init_views.shrink_to_fit();
		//cout << "NBV_Net_Labeler: share_data use_count is " << share_data.use_count() << endl;
		//cout << "NBV_Net_Labeler: view_space use_count is " << view_space.use_count() << endl;
		//cout << "NBV_Net_Labeler: percept use_count is " << percept.use_count() << endl;
	}

	NBV_Net_Labeler(shared_ptr<Share_Data>& _share_data, int _toward_state = 0, int _rotate_state = 0) {
		share_data = _share_data;
		toward_state = _toward_state;
		rotate_state = _rotate_state;
		object_is_ok_size = true;
		cout << "toward_state is " << toward_state << " , rotate_state is " << rotate_state << endl;

		if (share_data->is_shape_net) {
			/*
			if (pcl::io::loadPolygonFilePLY(share_data->model_path + "ShapeNet/" + share_data->name_of_pcd + ".ply", *share_data->mesh_ply) == -1) {
				cout << "Mesh not available. Please check if the file (or path) exists." << endl;
			}
			pcl::fromPCLPointCloud2(share_data->mesh_ply->cloud, *share_data->cloud_pcd);
			if (share_data->cloud_pcd->points.size() == 0) {
				cout << "points size wrong. Check." << endl;
			}
			*/
			if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(share_data->model_path + "ShapeNet/" + share_data->name_of_pcd + ".ply", *share_data->cloud_pcd) == -1) {
				cout << "Can not read 3d model file. Check." << endl;
			}
			cout << "points size is " << share_data->cloud_pcd->points.size() << endl;
		}
		else {
			//读取点云和mesh
			if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(share_data->pcd_file_path + share_data->name_of_pcd + ".pcd", *share_data->cloud_pcd) == -1) {
				cout << "Can not read 3d model file. Check." << endl;
			}
			if (pcl::io::loadPolygonFilePLY(share_data->ply_file_path + share_data->name_of_pcd + ".ply", *share_data->mesh_ply) == -1) {
				cout << "Mesh not available. Please check if the file (or path) exists." << endl;
			}
			share_data->mesh_data_offset = share_data->mesh_ply->cloud.data.size() / share_data->mesh_ply->cloud.width / share_data->mesh_ply->cloud.height;
			cout << "mesh field offset is " << share_data->mesh_data_offset << endl;
		}

		//旋转Z轴向上
		set<string> names_rotate;
		names_rotate.insert("Armadillo");
		names_rotate.insert("Asian_Dragon");
		names_rotate.insert("Dragon");
		names_rotate.insert("Stanford_Bunny");
		names_rotate.insert("Happy_Buddha");
		names_rotate.insert("Thai_Statue");
		//names_rotate.insert("Lucy");
		if (names_rotate.count(share_data->name_of_pcd) || share_data->is_shape_net) {
			pcl::transformPointCloud(*share_data->cloud_pcd, *share_data->cloud_pcd, share_data->get_toward_pose(4));
			//旋转mesh
			for (int i = 0; i < share_data->mesh_ply->cloud.data.size(); i += share_data->mesh_data_offset) {
				int arrayPosX = i + share_data->mesh_ply->cloud.fields[0].offset;
				int arrayPosY = i + share_data->mesh_ply->cloud.fields[1].offset;
				int arrayPosZ = i + share_data->mesh_ply->cloud.fields[2].offset;
				float X = 0.0;	float Y = 0.0;	float Z = 0.0;
				memcpy(&X, &share_data->mesh_ply->cloud.data[arrayPosX], sizeof(float));
				memcpy(&Y, &share_data->mesh_ply->cloud.data[arrayPosY], sizeof(float));
				memcpy(&Z, &share_data->mesh_ply->cloud.data[arrayPosZ], sizeof(float));
				pcl::PointCloud<pcl::PointXYZ> cloud_point;
				cloud_point.points.resize(1);
				cloud_point.points[0] = pcl::PointXYZ(X, Y, Z);
				pcl::transformPointCloud(cloud_point, cloud_point, share_data->get_toward_pose(4));
				memcpy(&share_data->mesh_ply->cloud.data[arrayPosX], &cloud_point.points[0].x, sizeof(float));
				memcpy(&share_data->mesh_ply->cloud.data[arrayPosY], &cloud_point.points[0].y, sizeof(float));
				memcpy(&share_data->mesh_ply->cloud.data[arrayPosZ], &cloud_point.points[0].z, sizeof(float));
				cloud_point.points.clear();
				cloud_point.points.shrink_to_fit();
			}
		}

		//初始化GT
		//旋转6个朝向之一
		pcl::transformPointCloud(*share_data->cloud_pcd, *share_data->cloud_pcd, share_data->get_toward_pose(toward_state));
		for (int i = 0; i < share_data->mesh_ply->cloud.data.size(); i += share_data->mesh_data_offset) {
			int arrayPosX = i + share_data->mesh_ply->cloud.fields[0].offset;
			int arrayPosY = i + share_data->mesh_ply->cloud.fields[1].offset;
			int arrayPosZ = i + share_data->mesh_ply->cloud.fields[2].offset;
			float X = 0.0;	float Y = 0.0;	float Z = 0.0;
			memcpy(&X, &share_data->mesh_ply->cloud.data[arrayPosX], sizeof(float));
			memcpy(&Y, &share_data->mesh_ply->cloud.data[arrayPosY], sizeof(float));
			memcpy(&Z, &share_data->mesh_ply->cloud.data[arrayPosZ], sizeof(float));
			pcl::PointCloud<pcl::PointXYZ> cloud_point;
			cloud_point.points.resize(1);
			cloud_point.points[0] = pcl::PointXYZ(X, Y, Z);
			pcl::transformPointCloud(cloud_point, cloud_point, share_data->get_toward_pose(toward_state));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosX], &cloud_point.points[0].x, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosY], &cloud_point.points[0].y, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosZ], &cloud_point.points[0].z, sizeof(float));
			cloud_point.points.clear();
			cloud_point.points.shrink_to_fit();
		}
		//旋转8个角度之一
		Eigen::Matrix3d rotation;
		rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
			Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
			Eigen::AngleAxisd(45 * rotate_state * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
		Eigen::Matrix4d T_pose(Eigen::Matrix4d::Identity(4, 4));
		T_pose(0, 0) = rotation(0, 0); T_pose(0, 1) = rotation(0, 1); T_pose(0, 2) = rotation(0, 2); T_pose(0, 3) = 0;
		T_pose(1, 0) = rotation(1, 0); T_pose(1, 1) = rotation(1, 1); T_pose(1, 2) = rotation(1, 2); T_pose(1, 3) = 0;
		T_pose(2, 0) = rotation(2, 0); T_pose(2, 1) = rotation(2, 1); T_pose(2, 2) = rotation(2, 2); T_pose(2, 3) = 0;
		T_pose(3, 0) = 0;			   T_pose(3, 1) = 0;			  T_pose(3, 2) = 0;			     T_pose(3, 3) = 1;
		pcl::transformPointCloud(*share_data->cloud_pcd, *share_data->cloud_pcd, T_pose);
		for (int i = 0; i < share_data->mesh_ply->cloud.data.size(); i += share_data->mesh_data_offset) {
			int arrayPosX = i + share_data->mesh_ply->cloud.fields[0].offset;
			int arrayPosY = i + share_data->mesh_ply->cloud.fields[1].offset;
			int arrayPosZ = i + share_data->mesh_ply->cloud.fields[2].offset;
			float X = 0.0;	float Y = 0.0;	float Z = 0.0;
			memcpy(&X, &share_data->mesh_ply->cloud.data[arrayPosX], sizeof(float));
			memcpy(&Y, &share_data->mesh_ply->cloud.data[arrayPosY], sizeof(float));
			memcpy(&Z, &share_data->mesh_ply->cloud.data[arrayPosZ], sizeof(float));
			pcl::PointCloud<pcl::PointXYZ> cloud_point;
			cloud_point.points.resize(1);
			cloud_point.points[0] = pcl::PointXYZ(X, Y, Z);
			pcl::transformPointCloud(cloud_point, cloud_point, T_pose);
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosX], &cloud_point.points[0].x, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosY], &cloud_point.points[0].y, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosZ], &cloud_point.points[0].z, sizeof(float));
			cloud_point.points.clear();
			cloud_point.points.shrink_to_fit();
		}
		//share_data->access_directory(share_data->save_path);
		//pcl::io::savePCDFile<pcl::PointXYZ>(share_data->save_path+to_string(toward_state)+"_"+ to_string(rotate_state) +".pcd", *cloud_pcd);

		//GT cloud
		share_data->cloud_ground_truth->is_dense = false;
		share_data->cloud_ground_truth->points.resize(share_data->cloud_pcd->points.size());
		share_data->cloud_ground_truth->width = share_data->cloud_pcd->points.size();
		share_data->cloud_ground_truth->height = 1;
		auto ptr = share_data->cloud_ground_truth->points.begin();
		auto p = share_data->cloud_pcd->points.begin();
		float unit = 1.0;
		if (!share_data->is_shape_net) {
			for (auto& ptr : share_data->cloud_pcd->points) {
				if (fabs(ptr.x) >= 10 || fabs(ptr.y) >= 10 || fabs(ptr.z) >= 10) {
					unit = 0.001;
					cout << "change unit from <mm> to <m>." << endl;
					break;
				}
			}
		}
		//检查物体大小，统一缩放为0.10m左右
		vector<Eigen::Vector3d> points;
		for (auto& ptr : share_data->cloud_pcd->points) {
			//cout<< "ptr: " << ptr.x << " " << ptr.y << " " << ptr.z << endl;
			Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
			points.push_back(pt);
		}
		Eigen::Vector3d object_center_world = Eigen::Vector3d(0, 0, 0);
		//计算点云质心
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
		cout<< "center original is " << object_center_world(0) << " " << object_center_world(1) << " " << object_center_world(2) << endl;
	
		//平移点云使得重心为0,0,0
		for (auto& ptr : share_data->cloud_pcd->points) {
			ptr.x = (ptr.x - object_center_world(0));
			ptr.y = (ptr.y - object_center_world(1));
			ptr.z = (ptr.z - object_center_world(2));
		}
		for (int i = 0; i < share_data->mesh_ply->cloud.data.size(); i += share_data->mesh_data_offset) {
			int arrayPosX = i + share_data->mesh_ply->cloud.fields[0].offset;
			int arrayPosY = i + share_data->mesh_ply->cloud.fields[1].offset;
			int arrayPosZ = i + share_data->mesh_ply->cloud.fields[2].offset;
			float X = 0.0;	float Y = 0.0;	float Z = 0.0;
			memcpy(&X, &share_data->mesh_ply->cloud.data[arrayPosX], sizeof(float));
			memcpy(&Y, &share_data->mesh_ply->cloud.data[arrayPosY], sizeof(float));
			memcpy(&Z, &share_data->mesh_ply->cloud.data[arrayPosZ], sizeof(float));
			X = float(X - object_center_world(0));
			Y = float(Y - object_center_world(1));
			Z = float(Z - object_center_world(2));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosX], &X, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosY], &Y, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosZ], &Z, sizeof(float));
		}
		//再次计算中心
		points.clear();
		points.shrink_to_fit();
		for (auto& ptr : share_data->cloud_pcd->points) {
			Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
			points.push_back(pt);
		}
		object_center_world = Eigen::Vector3d(0, 0, 0);
		//计算点云质心
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
		if (object_center_world.norm() > 1e-6) {
			cout << "error with move to centre." << endl;
		}

		//计算最远点
		double predicted_size = 0.0;
		for (auto& ptr : points) {
			predicted_size = max(predicted_size, (object_center_world - ptr).norm());
		}
		predicted_size *= 17.0/16.0;
		
		double scale = 1.0;
		if (share_data->mp_scale.find(share_data->name_of_pcd)!= share_data->mp_scale.end()) {
			scale = (predicted_size - share_data->mp_scale[share_data->name_of_pcd]) / predicted_size;
			cout << "object " << share_data->name_of_pcd << " large. change scale " << predicted_size << " to about " << predicted_size - share_data->mp_scale[share_data->name_of_pcd] << " m." << endl;
		}
		else {
			cout << "object " << share_data->name_of_pcd << " size is " << predicted_size << " m." << endl;
		}

		//固定大小为0.10m
		//scale = 0.10 / predicted_size;
		//cout << "object " << share_data->name_of_pcd << " change scale " << predicted_size << " to about " << 0.10 << " m." << endl;

		//释放内存
		points.clear();
		points.shrink_to_fit();

		if (share_data->is_shape_net) {
			double random_size = -1;
			share_data->access_directory(share_data->gt_path);
			ifstream size_reader(share_data->gt_path + "/size.txt");
			if (size_reader.is_open()) {
				size_reader >> random_size;
				size_reader.close();
				if (random_size < 0) {
					cout << "no size. Skip." << endl;
					object_is_ok_size = false;
					return;
				}
			}
			else {
				size_reader.close();
				random_size = 0.075;
				double object_rate = -1;
				int test_times = 0;
				do{
					random_size = get_random_coordinate(random_size, 0.115);
					cout << "random size is " << random_size << endl;
					//从cloud_pcd复制一个点云，并将尺寸置为random_size
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_test_size(new pcl::PointCloud<pcl::PointXYZRGB>);
					for (auto& ptr : share_data->cloud_pcd->points) {
						pcl::PointXYZRGB pt;
						pt.x = ptr.x * random_size / predicted_size;
						pt.y = ptr.y * random_size / predicted_size;
						pt.z = ptr.z * random_size / predicted_size;
						pt.r = ptr.r;
						pt.g = ptr.g;
						pt.b = ptr.b;
						cloud_test_size->push_back(pt);
					}
					pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Render"));
					viewer->setBackgroundColor(255, 255, 255);
					viewer->initCameraParameters();
					viewer->addPointCloud<pcl::PointXYZRGB>(cloud_test_size, "cloud_test_size");
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, share_data->points_size_cloud, "cloud_test_size");

					share_data->access_directory(share_data->pre_path);

					ifstream fin_sphere(share_data->viewspace_path + "5.txt");
					for (int i = 0; i < 5; i++) {
						Eigen::Vector3d init_pos;
						fin_sphere >> init_pos(0) >> init_pos(1) >> init_pos(2);
						init_pos = init_pos / init_pos.norm() * share_data->view_space_radius + object_center_world;
						View view(init_pos);
						Eigen::Matrix4d view_pose_world;
						view.get_next_camera_pos(share_data->now_camera_pose_world, object_center_world);
						view_pose_world = (share_data->now_camera_pose_world * view.pose.inverse()).eval();
						//渲染
						Eigen::Matrix3f intrinsics;
						intrinsics << share_data->color_intrinsics.fx, 0, share_data->color_intrinsics.ppx,
							0, share_data->color_intrinsics.fy, share_data->color_intrinsics.ppy,
							0, 0, 1;
						Eigen::Matrix4f extrinsics = view_pose_world.cast<float>();
						viewer->setCameraParameters(intrinsics, extrinsics);
						pcl::visualization::Camera cam;
						viewer->getCameraParameters(cam);
						cam.window_size[0] = share_data->color_intrinsics.width;
						cam.window_size[1] = share_data->color_intrinsics.height;
						viewer->setCameraParameters(cam);
						viewer->spinOnce(100);
						viewer->saveScreenshot(share_data->gt_path + "/rgb_size_test_" + to_string(i) + ".png");
					}

					//图片中不是白色的像素点的个数
					double now_object_rate = 0.0;
					for (int i = 0; i < 5; i++) {
						cv::Mat img_size_test = cv::imread(share_data->gt_path + "/rgb_size_test_" + to_string(i) + ".png");
						int count_object = 0;
						for (int i = 0; i < img_size_test.rows; i++) {
							for (int j = 0; j < img_size_test.cols; j++) {
								if (img_size_test.at<cv::Vec3b>(i, j)[0] != 255 || img_size_test.at<cv::Vec3b>(i, j)[1] != 255 || img_size_test.at<cv::Vec3b>(i, j)[2] != 255) {
									count_object++;
								}
							}
						}
						now_object_rate += (double)count_object / (img_size_test.rows * img_size_test.cols);

						img_size_test.release();
						remove((share_data->gt_path + "/rgb_size_test_" + to_string(i) + ".png").c_str());
					}
					now_object_rate /= 5;
					cout << "now object rate is " << now_object_rate << endl;

					viewer->removePointCloud("cloud_test_size");
					viewer->close();
					viewer.reset();
					cloud_test_size->points.clear();
					cloud_test_size->points.shrink_to_fit();
					cloud_test_size.reset();

					test_times++;
					object_rate = now_object_rate;

				} while (object_rate <= share_data->object_pixel_rate && test_times <= 5);
				if (test_times <= 5) {
					ofstream fout_radom_size(share_data->gt_path + "/size.txt");
					fout_radom_size << random_size;
					fout_radom_size.close();
				}
				else {
					object_is_ok_size = false;
					ofstream fout_radom_size(share_data->gt_path + "/size.txt");
					fout_radom_size << -1;
					fout_radom_size.close();
					return;
				}
			}
			scale = random_size / predicted_size;
			//unit = 1.0 / 1024.0;
			cout << "shapenet object " << share_data->name_of_pcd << " random size is " << random_size << " m." << endl;
		}

		//动态分辨率
		double predicted_octomap_resolution = scale * predicted_size * 2.0 / 32.0;
		cout << "choose octomap_resolution: " << predicted_octomap_resolution << " m." << endl;
		share_data->octomap_resolution = predicted_octomap_resolution;
		share_data->octo_model = make_shared<octomap::ColorOcTree>(share_data->octomap_resolution);
		share_data->GT_sample = make_shared<octomap::ColorOcTree>(share_data->octomap_resolution);
		//测试BBX尺寸
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++)
				for (int k = 0; k < 32; k++)
				{
					double x = object_center_world(0) * scale * unit - scale * predicted_size + share_data->octomap_resolution * i;
					double y = object_center_world(1) * scale * unit - scale * predicted_size + share_data->octomap_resolution * j;
					double z = object_center_world(2) * scale * unit - scale * predicted_size + share_data->octomap_resolution * k;
					share_data->GT_sample->setNodeValue(x, y, z, share_data->GT_sample->getProbMissLog(), true); //初始化概率0
					//cout << x << " " << y << " " << z << endl;
				}

		//转换mesh
		for (int i = 0; i < share_data->mesh_ply->cloud.data.size(); i += share_data->mesh_data_offset) {
			int arrayPosX = i + share_data->mesh_ply->cloud.fields[0].offset;
			int arrayPosY = i + share_data->mesh_ply->cloud.fields[1].offset;
			int arrayPosZ = i + share_data->mesh_ply->cloud.fields[2].offset;
			float X = 0.0;	float Y = 0.0;	float Z = 0.0;
			memcpy(&X, &share_data->mesh_ply->cloud.data[arrayPosX], sizeof(float));
			memcpy(&Y, &share_data->mesh_ply->cloud.data[arrayPosY], sizeof(float));
			memcpy(&Z, &share_data->mesh_ply->cloud.data[arrayPosZ], sizeof(float));
			X = float(X * scale * unit);
			Y = float(Y * scale * unit);
			Z = float(Z * scale * unit);
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosX], &X, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosY], &Y, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosZ], &Z, sizeof(float));
		}
		

		//转换点云
		//double min_z = 0;
		double min_z = object_center_world(2) * scale * unit;
		for (int i = 0; i < share_data->cloud_pcd->points.size(); i++, p++)
		{
			//RGB点云
			(*ptr).x = (*p).x * scale * unit;
			(*ptr).y = (*p).y * scale * unit;
			(*ptr).z = (*p).z * scale * unit;
			(*ptr).r = (*p).r;
			(*ptr).g = (*p).g;
			(*ptr).b = (*p).b;
			//GT OctoMap插入点云
			octomap::OcTreeKey key;  bool key_have = share_data->ground_truth_model->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
			if (key_have) {
				octomap::ColorOcTreeNode* voxel = share_data->ground_truth_model->search(key);
				if (voxel == NULL) {
					share_data->ground_truth_model->setNodeValue(key, share_data->ground_truth_model->getProbHitLog(), true);
					share_data->ground_truth_model->integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
				}
				voxel = NULL;
			}
			min_z = min(min_z, (double)(*ptr).z);
			//GT_sample OctoMap插入点云
			octomap::OcTreeKey key_sp;  bool key_have_sp = share_data->GT_sample->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key_sp);
			if (key_have_sp) {
				octomap::ColorOcTreeNode* voxel_sp = share_data->GT_sample->search(key_sp);
				//if (voxel_sp == NULL) {
					share_data->GT_sample->setNodeValue(key_sp, share_data->GT_sample->getProbHitLog(), true);
					share_data->GT_sample->integrateNodeColor(key_sp, 255, 0, 0);
				//}
				voxel_sp = NULL;
			}
			ptr++;
		}
		//记录桌面
		share_data->min_z_table = min_z - share_data->ground_truth_resolution;

		//share_data->access_directory(share_data->save_path);
		share_data->ground_truth_model->updateInnerOccupancy();
		//share_data->ground_truth_model->write(share_data->save_path + "/GT.ot");
		//GT_sample_voxels
		share_data->GT_sample->updateInnerOccupancy();
		//share_data->GT_sample->write(share_data->save_path + "/GT_sample.ot");
		share_data->init_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->GT_sample->begin_leafs(), end = share_data->GT_sample->end_leafs(); it != end; ++it) {
			share_data->init_voxels++;
		}
		//cout << "Map_GT_sample has voxels " << share_data->init_voxels << endl;
		//if (share_data->init_voxels != 32768) cout << "WARNING! BBX small." << endl;
		//ofstream fout(share_data->save_path + "/GT_size.txt");
		//fout << scale * predicted_size << endl;

		share_data->full_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			share_data->full_voxels++;
		}

		//初始化viewspace
		view_space = make_shared<View_Space>(share_data);

		//show mesh space
		if (share_data->show) {
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("mesh"));
			viewer1->setBackgroundColor(255, 255, 255);
			viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			double max_z = 0;
			int index_z = 0;
			for (int i = 0; i < view_space->views.size(); i++) {
				Eigen::Vector4d X(0.01, 0, 0, 1);
				Eigen::Vector4d Y(0, 0.01, 0, 1);
				Eigen::Vector4d Z(0, 0, 0.01, 1);
				Eigen::Vector4d O(0, 0, 0, 1);
				view_space->views[i].get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), object_center_world);
				Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * view_space->views[i].pose.inverse()).eval();
				//cout << view_pose_world << endl;
				X = view_pose_world * X;
				Y = view_pose_world * Y;
				Z = view_pose_world * Z;
				O = view_pose_world * O;
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "X" + to_string(i));
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Y" + to_string(i));
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Z" + to_string(i));

				if (view_space->views[i].init_pos(1) > max_z) {
					max_z = view_space->views[i].init_pos(2);
					index_z = i;
				}
			}
			cout << "z_max_index is " << index_z << endl;
			viewer1->addPolygonMesh(*share_data->mesh_ply, "mesh_ply");
			view_space->add_bbx_to_cloud(viewer1);
			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
			viewer1->removeAllCoordinateSystems();
			viewer1->removeAllPointClouds();
			viewer1->removeAllShapes();
			viewer1->close();
			viewer1.reset();
		}

		//相机类初始化
		percept = make_shared<Perception_3D>(share_data);
		
		//srand(time(0));
		
	}

	void get_cover_view_cases() {
		double now_time = clock();
		//从球面生成半球面coverage视点空间
		for (int i = 0; i <= 200; i++) {
		//for (int i = 1080; i <= 1080; i++) {
			vector<Eigen::Vector4d> view_points_uniform;
			ifstream fin_sphere(share_data->orginalviews_path + to_string(i) + ".txt");
			int num,id; 
			double x, y, z, dis, angel;
			fin_sphere >> num >> dis >> angel;
			for (int j = 0; j < i; j++) {
				fin_sphere >> id >> x >> y >> z;
				double r = sqrt(x * x + y * y + z * z);
				view_points_uniform.push_back(Eigen::Vector4d(x / r, y / r, z / r, 1.0));
			}
			for (int k = 0; k < i; k++) {
				//旋转视点空间，取第k个视点旋转为坐标0 0 1
				Eigen::Vector3d Z = Eigen::Vector3d(view_points_uniform[k](0), view_points_uniform[k](1), view_points_uniform[k](2));
				Eigen::Vector3d X = Eigen::Vector3d(1, 1, -(Z(0) + Z(1)) / Z(2)); X = X.normalized();
				Eigen::Vector3d Y = Z.cross(X); Y = Y.normalized();
				Eigen::Matrix4d R(4, 4);
				R(0, 0) = X(0); R(0, 1) = Y(0); R(0, 2) = Z(0); R(0, 3) = 0;
				R(1, 0) = X(1); R(1, 1) = Y(1); R(1, 2) = Z(1); R(1, 3) = 0;
				R(2, 0) = X(2); R(2, 1) = Y(2); R(2, 2) = Z(2); R(2, 3) = 0;
				R(3, 0) = 0;	R(3, 1) = 0;	R(3, 2) = 0;	R(3, 3) = 1;
				vector<Eigen::Vector3d> out_view_points;
				for (int j = 0; j < i; j++) {
					view_points_uniform[j] = R.inverse() * view_points_uniform[j];
					if (view_points_uniform[j](2) >= 0) out_view_points.push_back(Eigen::Vector3d(view_points_uniform[j](0), view_points_uniform[j](1), view_points_uniform[j](2)));
				}
				//保存视点空间
				ifstream fin_vs(share_data->viewspace_path + to_string(out_view_points.size()) + ".txt");
				if (!fin_vs.is_open()) {
					fin_vs.close();
					ofstream fout_vs(share_data->viewspace_path + to_string(out_view_points.size()) + ".txt");
					for (int j = 0; j < out_view_points.size(); j++) {
						fout_vs << out_view_points[j](0) << ' ' << out_view_points[j](1) << ' ' << out_view_points[j](2) << '\n';
					}
					fout_vs.close();
				}
				else {
					vector<Eigen::Vector3d> pre_view_points;
					double x, y, z;
					while (fin_vs >> x >> y >> z) {
						pre_view_points.push_back(Eigen::Vector3d(x, y, z));
					}
					fin_vs.close();
					double pre_dis = 0, dis = 0;
					for (int x = 0; x < out_view_points.size(); x++)
						for (int y = x + 1; y < out_view_points.size(); y++) {
							pre_dis += (pre_view_points[x] - pre_view_points[y]).norm();
							dis += (out_view_points[x] - out_view_points[y]).norm();
						}
					if (dis >= pre_dis) {
						ofstream fout_vs(share_data->viewspace_path + to_string(out_view_points.size()) + ".txt");
						for (int j = 0; j < out_view_points.size(); j++) {
							fout_vs << out_view_points[j](0) << ' ' << out_view_points[j](1) << ' ' << out_view_points[j](2) << '\n';
						}
						fout_vs.close();
					}
				}
			}
		}

		cout << "view cases get with executed time " << clock() - now_time << " ms." << endl;
	}

	void get_novel_view_cases() {
		double now_time = clock();

		//读取coverage视点空间
		set<pair<double, pair<double, double>>> view_check_set;
		int num_of_coverage_views = 0;
		for (int i = 3; i <= 100; i++) {
			ifstream fin_sphere(share_data->viewspace_path + to_string(i) + ".txt");
			vector<Eigen::Vector3d> view_points;
			pair<double, pair<double, double>> temp;
			for (int j = 0; j < i; j++) {
				double x, y, z;
				fin_sphere >> x >> y >> z;
				double r = sqrt(x * x + y * y + z * z);
				view_check_set.insert(make_pair(x / r, make_pair(y / r, z / r)));
				view_points.push_back(Eigen::Vector3d(x / r, y / r, z / r));
				num_of_coverage_views++;
			}
			//check viewspace
			if (share_data->show) {
				continue;
				//if (i != 3 && i != 5 && i != 10 && i != 25 && i != 50 && i != 100) continue;
				pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("points_on_sphere"));
				viewer->setBackgroundColor(255, 255, 255);
				viewer->addCoordinateSystem(0.1);
				viewer->initCameraParameters();
				for (int i = 0; i < view_points.size(); i++) {
					Eigen::Vector4d X(0.1, 0, 0, 1);
					Eigen::Vector4d Y(0, 0.1, 0, 1);
					Eigen::Vector4d Z(0, 0, 0.1, 1);
					Eigen::Vector4d O(0, 0, 0, 1);
					View temp_view(view_points[i]);
					temp_view.get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * temp_view.pose.inverse()).eval();
					//cout << view_pose_world << endl;
					X = view_pose_world * X;
					Y = view_pose_world * Y;
					Z = view_pose_world * Z;
					O = view_pose_world * O;
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i));
				}
				viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");
				while (!viewer->wasStopped())
				{
					viewer->spinOnce(100);
					boost::this_thread::sleep(boost::posix_time::microseconds(100000));
				}
				viewer.reset();
			}
			view_points.clear();
			view_points.shrink_to_fit();
		}
		cout<< "num_of_coverage_views: " << num_of_coverage_views << endl;
		cout<< "view_check_set.size(): " << view_check_set.size() << endl;
		if(num_of_coverage_views != view_check_set.size()) cout<< "WARNING! num_of_coverage_views != view_check_set.size()" << endl;

		//均匀半球面采样100个点，用于training
		share_data->access_directory(share_data->pre_path);
		ifstream fin_training_novel_views(share_data->pre_path + "/novel_train_views.txt");
		if (!fin_training_novel_views.is_open()) {
			fin_training_novel_views.close();

			vector<Eigen::Vector3d> novel_training_views;
			for (int k = 0; k < 10000; k++) {
				vector<Eigen::Vector3d> novel_training_views_try;
				while (novel_training_views_try.size() != share_data->num_of_novel_test_views) {
					double x = get_random_coordinate(-1, 1);
					double y = get_random_coordinate(-1, 1);
					double z = get_random_coordinate(-1, 1);
					if (z < 0) continue;
					double r = sqrt(x * x + y * y + z * z);
					if (view_check_set.count(make_pair(x / r, make_pair(y / r, z / r)))) continue;
					novel_training_views_try.push_back(Eigen::Vector3d(x / r, y / r, z / r));
				}
				if(k==0) novel_training_views = novel_training_views_try;
				else {
					double pre_dis = 0, dis = 0;
					for (int x = 0; x < novel_training_views.size(); x++)
						for (int y = x + 1; y < novel_training_views.size(); y++) {
							pre_dis += (novel_training_views[x] - novel_training_views[y]).norm();
							dis += (novel_training_views_try[x] - novel_training_views_try[y]).norm();
						}
					double pre_dis_weighted = pre_dis, dis_weighted = dis;
					for (int x = 0; x < novel_training_views.size(); x++) {
						if (novel_training_views[x](2) >= 0.8) pre_dis_weighted += pre_dis / novel_training_views.size();
						if (novel_training_views_try[x](2) >= 0.8) dis_weighted += dis / novel_training_views.size();
					}
					if (dis_weighted >= pre_dis_weighted) {
						novel_training_views = novel_training_views_try;
					}
				}
			}

			ofstream fout_training_novel_views(share_data->pre_path + "/novel_train_views.txt");
			for(int k=0;k< novel_training_views.size();k++){
				view_check_set.insert(make_pair(novel_training_views[k](0), make_pair(novel_training_views[k](1), novel_training_views[k](2))));
				fout_training_novel_views << novel_training_views[k](0) << " " << novel_training_views[k](1) << " " << novel_training_views[k](2) << endl;
			}
			if (share_data->show) {
				pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("points_on_sphere"));
				viewer->setBackgroundColor(255, 255, 255);
				viewer->addCoordinateSystem(0.1);
				viewer->initCameraParameters();
				for (int i = 0; i < novel_training_views.size(); i++) {
					Eigen::Vector4d X(0.01, 0, 0, 1);
					Eigen::Vector4d Y(0, 0.01, 0, 1);
					Eigen::Vector4d Z(0, 0, 0.01, 1);
					Eigen::Vector4d O(0, 0, 0, 1);
					View temp_view(novel_training_views[i]);
					temp_view.get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * temp_view.pose.inverse()).eval();
					//cout << view_pose_world << endl;
					X = view_pose_world * X;
					Y = view_pose_world * Y;
					Z = view_pose_world * Z;
					O = view_pose_world * O;
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "X" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Y" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Z" + to_string(i));
				}
				viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");
				while (!viewer->wasStopped())
				{
					viewer->spinOnce(100);
					boost::this_thread::sleep(boost::posix_time::microseconds(100000));
				}
				viewer.reset();
			}
			//释放内存
			novel_training_views.clear();
			novel_training_views.shrink_to_fit();
		}
		else {
			double x, y, z;
			while (fin_training_novel_views >> x >> y >> z) {
				view_check_set.insert(make_pair(x, make_pair(y, z)));
			}
			fin_training_novel_views.close();
		}

		//均匀半球面采样100个点，用于novel testing
		ifstream fin_test_novel_views(share_data->pre_path + "/novel_test_views.txt");
		if (!fin_test_novel_views.is_open()) {
			fin_test_novel_views.close();

			vector<Eigen::Vector3d> novel_test_views;
			for (int k = 0; k < 10000; k++) {
				vector<Eigen::Vector3d> novel_test_views_try;
				while (novel_test_views_try.size() != share_data->num_of_novel_test_views) {
					double x = get_random_coordinate(-1, 1);
					double y = get_random_coordinate(-1, 1);
					double z = get_random_coordinate(-1, 1);
					if (z < 0) continue;
					double r = sqrt(x * x + y * y + z * z);
					if (view_check_set.count(make_pair(x / r, make_pair(y / r, z / r)))) continue;
					novel_test_views_try.push_back(Eigen::Vector3d(x / r, y / r, z / r));
				}
				if (k == 0) novel_test_views = novel_test_views_try;
				else {
					double pre_dis = 0, dis = 0;
					for (int x = 0; x < novel_test_views.size(); x++)
						for (int y = x + 1; y < novel_test_views.size(); y++) {
							pre_dis += (novel_test_views[x] - novel_test_views[y]).norm();
							dis += (novel_test_views_try[x] - novel_test_views_try[y]).norm();
						}
					double pre_dis_weighted = pre_dis, dis_weighted = dis;
					for (int x = 0; x < novel_test_views.size(); x++) {
						if (novel_test_views[x](2) >= 0.8) pre_dis_weighted += pre_dis / novel_test_views.size();
						if (novel_test_views_try[x](2) >= 0.8) dis_weighted += dis / novel_test_views.size();
					}
					if (dis_weighted >= pre_dis_weighted) {
						novel_test_views = novel_test_views_try;
					}
				}
			}

			ofstream fout_test_novel_views(share_data->pre_path + "/novel_test_views.txt");
			for (int k = 0; k < novel_test_views.size(); k++) {
				view_check_set.insert(make_pair(novel_test_views[k](0), make_pair(novel_test_views[k](1), novel_test_views[k](2))));
				fout_test_novel_views << novel_test_views[k](0) << " " << novel_test_views[k](1) << " " << novel_test_views[k](2) << endl;
			}

			if (share_data->show) {
				pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("points_on_sphere"));
				viewer->setBackgroundColor(255, 255, 255);
				viewer->addCoordinateSystem(0.1);
				viewer->initCameraParameters();
				for (int i = 0; i < novel_test_views.size(); i++) {
					Eigen::Vector4d X(0.01, 0, 0, 1);
					Eigen::Vector4d Y(0, 0.01, 0, 1);
					Eigen::Vector4d Z(0, 0, 0.01, 1);
					Eigen::Vector4d O(0, 0, 0, 1);
					View temp_view(novel_test_views[i]);
					temp_view.get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * temp_view.pose.inverse()).eval();
					//cout << view_pose_world << endl;
					X = view_pose_world * X;
					Y = view_pose_world * Y;
					Z = view_pose_world * Z;
					O = view_pose_world * O;
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "X" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Y" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Z" + to_string(i));
				}
				viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");
				while (!viewer->wasStopped())
				{
					viewer->spinOnce(100);
					boost::this_thread::sleep(boost::posix_time::microseconds(100000));
				}
				viewer.reset();
			}
			//释放内存
			novel_test_views.clear();
			novel_test_views.shrink_to_fit();
		}

		cout << "view cases get with executed time " << clock() - now_time << " ms." << endl;
	}

	int get_train_test_novel() {
		double now_time = clock();

		//json root
		Json::Value root;
		root["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root["fl_x"] = share_data->color_intrinsics.fx;
		root["fl_y"] = share_data->color_intrinsics.fy;
		root["k1"] = share_data->color_intrinsics.coeffs[0];
		root["k2"] = share_data->color_intrinsics.coeffs[1];
		root["k3"] = share_data->color_intrinsics.coeffs[2];
		root["p1"] = share_data->color_intrinsics.coeffs[3];
		root["p2"] = share_data->color_intrinsics.coeffs[4];
		root["cx"] = share_data->color_intrinsics.ppx;
		root["cy"] = share_data->color_intrinsics.ppy;
		root["w"] = share_data->color_intrinsics.width;
		root["h"] = share_data->color_intrinsics.height;
		root["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root["scale"] = 0.5 / share_data->predicted_size;
		root["offset"][0] = 0.5 + share_data->object_center_world(2);
		root["offset"][1] = 0.5 + share_data->object_center_world(0);
		root["offset"][2] = 0.5 + share_data->object_center_world(1);

		//读取100个球面随机视点
		ifstream fin_train_novel_views(share_data->pre_path + "/novel_train_views.txt");
		vector<View> novel_train_views;
		double x, y, z;
		while (fin_train_novel_views >> x >> y >> z) {
			Eigen::Vector3d novel_view_pos(x, y, z);
			double scale = 1.0 / novel_view_pos.norm() * share_data->view_space_radius;
			View novel_view(Eigen::Vector3d(x * scale + share_data->object_center_world(0), y * scale + share_data->object_center_world(1), z * scale + share_data->object_center_world(2)));
			novel_train_views.push_back(novel_view);
		}
		//每个测试视点成像并写入文件
		share_data->clouds.clear();
		for (int i = 0; i < novel_train_views.size(); i++) {
			//get point cloud
			//percept->precept(novel_train_views[i]);
			share_data->access_directory(share_data->gt_path + "/novel_train/");
			//get rgb image
			percept->render(novel_train_views[i], i, "/novel_train");
			cv::Mat rgb_image = cv::imread(share_data->gt_path + "/novel_train/rgb_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
			cv::Mat rgb_image_alpha, rgb_image_alpha_clip;
			convertToAlpha(rgb_image, rgb_image_alpha);
			rgb_image_alpha_clip = rgb_image_alpha.clone();
			cv::flip(rgb_image_alpha_clip, rgb_image_alpha_clip, -1);
			cv::imwrite(share_data->gt_path + "/novel_train/rgbaClip_" + to_string(i) + ".png", rgb_image_alpha_clip);
			remove((share_data->gt_path + "/novel_train/rgb_" + to_string(i) + ".png").c_str());
		}

		//所有测试视点
		Json::Value root_case_train_views(root);
		for (int i = 0; i < novel_train_views.size(); i++) {
			Json::Value view_image;
			view_image["file_path"] = "novel_train/rgbaClip_" + to_string(i) + ".png";
			Json::Value transform_matrix;
			for (int k = 0; k < 4; k++) {
				Json::Value row;
				for (int l = 0; l < 4; l++) {
					novel_train_views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = share_data->now_camera_pose_world * novel_train_views[i].pose.inverse();
					//x,y,z->y,z,x
					Eigen::Matrix4d pose;
					pose << 0, 0, 1, 0,
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;
					//x,y,z->x,-y,-z
					Eigen::Matrix4d pose_1;
					pose_1 << 1, 0, 0, 0,
						0, -1, 0, 0,
						0, 0, -1, 0,
						0, 0, 0, 1;
					view_pose_world = pose * view_pose_world * pose_1;
					row.append(view_pose_world(k, l));
				}
				transform_matrix.append(row);
			}
			view_image["transform_matrix"] = transform_matrix;
			root_case_train_views["frames"].append(view_image);
		}
		Json::StyledWriter writer_train_views;
		ofstream fout_train_views(share_data->gt_path + "/novel_train_views.json");
		fout_train_views << writer_train_views.write(root_case_train_views);
		fout_train_views.close();

		//释放内存
		novel_train_views.clear();
		novel_train_views.shrink_to_fit();

		cout << "images get with executed time " << clock() - now_time << " ms." << endl;
		now_time = clock();

		//读取100个球面随机视点
		ifstream fin_test_novel_views(share_data->pre_path + "/novel_test_views.txt");
		vector<View> novel_test_views;
		//double x, y, z;
		while (fin_test_novel_views >> x >> y >> z) {
			Eigen::Vector3d novel_view_pos(x, y, z);
			double scale = 1.0 / novel_view_pos.norm() * share_data->view_space_radius;
			View novel_view(Eigen::Vector3d(x * scale + share_data->object_center_world(0), y * scale + share_data->object_center_world(1), z * scale + share_data->object_center_world(2)));
			novel_test_views.push_back(novel_view);
		}
		//每个测试视点成像并写入文件
		share_data->clouds.clear();
		for (int i = 0; i < novel_test_views.size(); i++) {
			//get point cloud
			//percept->precept(novel_test_views[i]);
			share_data->access_directory(share_data->gt_path + "/novel_test/");
			//get rgb image
			percept->render(novel_test_views[i], i, "/novel_test");
			cv::Mat rgb_image = cv::imread(share_data->gt_path + "/novel_test/rgb_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
			cv::Mat rgb_image_alpha, rgb_image_alpha_clip;
			convertToAlpha(rgb_image, rgb_image_alpha);
			rgb_image_alpha_clip = rgb_image_alpha.clone();
			cv::flip(rgb_image_alpha_clip, rgb_image_alpha_clip, -1);
			cv::imwrite(share_data->gt_path + "/novel_test/rgbaClip_" + to_string(i) + ".png", rgb_image_alpha_clip);
			remove((share_data->gt_path  + "/novel_test/rgb_" + to_string(i) + ".png").c_str());
		}

		//所有测试视点
		Json::Value root_case_test_views(root);
		for (int i = 0; i < novel_test_views.size(); i++) {
			Json::Value view_image;
			view_image["file_path"] = "novel_test/rgbaClip_" + to_string(i) + ".png";
			Json::Value transform_matrix;
			for (int k = 0; k < 4; k++) {
				Json::Value row;
				for (int l = 0; l < 4; l++) {
					novel_test_views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = share_data->now_camera_pose_world * novel_test_views[i].pose.inverse();
					//x,y,z->y,z,x
					Eigen::Matrix4d pose;
					pose << 0, 0, 1, 0,
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;
					//x,y,z->x,-y,-z
					Eigen::Matrix4d pose_1;
					pose_1 << 1, 0, 0, 0,
						0, -1, 0, 0,
						0, 0, -1, 0,
						0, 0, 0, 1;
					view_pose_world = pose * view_pose_world * pose_1;
					row.append(view_pose_world(k, l));
				}
				transform_matrix.append(row);
			}
			view_image["transform_matrix"] = transform_matrix;
			root_case_test_views["frames"].append(view_image);
		}
		Json::StyledWriter writer_novel_views;
		ofstream fout_novel_views(share_data->gt_path + "/novel_test_views.json");
		fout_novel_views << writer_novel_views.write(root_case_test_views);
		fout_novel_views.close();

		//释放内存
		novel_test_views.clear();
		novel_test_views.shrink_to_fit();

		cout << "images get with executed time " << clock() - now_time << " ms." << endl;

		return 0;
	}

	int get_coverage() {
		double now_time = clock();
		//json root
		Json::Value root;
		root["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root["fl_x"] = share_data->color_intrinsics.fx;
		root["fl_y"] = share_data->color_intrinsics.fy;
		root["k1"] = share_data->color_intrinsics.coeffs[0];
		root["k2"] = share_data->color_intrinsics.coeffs[1];
		root["k3"] = share_data->color_intrinsics.coeffs[2];
		root["p1"] = share_data->color_intrinsics.coeffs[3];
		root["p2"] = share_data->color_intrinsics.coeffs[4];
		root["cx"] = share_data->color_intrinsics.ppx;
		root["cy"] = share_data->color_intrinsics.ppy;
		root["w"] = share_data->color_intrinsics.width;
		root["h"] = share_data->color_intrinsics.height;
		root["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root["scale"] = 0.5 / share_data->predicted_size;
		root["offset"][0] = 0.5 + share_data->object_center_world(2);
		root["offset"][1] = 0.5 + share_data->object_center_world(0);
		root["offset"][2] = 0.5 + share_data->object_center_world(1);
		
		//每个视点成像并写入文件
		for (int i = 0; i < share_data->num_of_views; i++) {
			//get point cloud
			//percept->precept(view_space->views[i]);
			share_data->access_directory(share_data->gt_path + "/" + to_string(share_data->num_of_views));
			//get rgb image
			percept->render(view_space->views[i], i, "/" + to_string(share_data->num_of_views));
			cv::Mat rgb_image = cv::imread(share_data->gt_path + "/" + to_string(share_data->num_of_views) + "/rgb_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
			cv::Mat rgb_image_alpha, rgb_image_alpha_clip;
			convertToAlpha(rgb_image, rgb_image_alpha);
			//cv::imwrite(share_data->gt_path + "/" + to_string(share_data->num_of_views) + "/rgba_" + to_string(i) + ".png", rgb_image_alpha);
			rgb_image_alpha_clip = rgb_image_alpha.clone();
			cv::flip(rgb_image_alpha_clip, rgb_image_alpha_clip, -1);
			cv::imwrite(share_data->gt_path + "/" + to_string(share_data->num_of_views) + "/rgbaClip_" + to_string(i) + ".png", rgb_image_alpha_clip);
			remove((share_data->gt_path + "/" + to_string(share_data->num_of_views) + "/rgb_" + to_string(i) + ".png").c_str());
			//get json
			Json::Value view_image;
			view_image["file_path"] = to_string(share_data->num_of_views) + "/rgbaClip_" + to_string(i) + ".png";
			Json::Value transform_matrix;
			for (int k = 0; k < 4; k++) {
				Json::Value row;
				for (int l = 0; l < 4; l++) {
					view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = share_data->now_camera_pose_world * view_space->views[i].pose.inverse();
					//x,y,z->y,z,x
					Eigen::Matrix4d pose;
					pose << 0, 0, 1, 0,
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;
					//x,y,z->x,-y,-z
					Eigen::Matrix4d pose_1;
					pose_1 << 1, 0, 0, 0,
						0, -1, 0, 0,
						0, 0, -1, 0,
						0, 0, 0, 1;
					view_pose_world = pose * view_pose_world * pose_1;
					row.append(view_pose_world(k, l));
				}
				transform_matrix.append(row);
			}
			view_image["transform_matrix"] = transform_matrix;
			root["frames"].append(view_image);
		}
		Json::StyledWriter writer;
		ofstream fout(share_data->gt_path + "/" + to_string(share_data->num_of_views) + ".json");
		fout << writer.write(root);
		fout.close();

		cout << "images get with executed time " << clock() - now_time << " ms." << endl;

		return 0;
	}

	int train_by_instantNGP(string trian_json_file, string test_json_file = "100", bool nbv_test = false, int ensemble_id = -1) {
		double now_time = clock();
		//使用命令行训练
		ofstream fout_py(share_data->instant_ngp_path + "interact/run_with_c++.py");

		fout_py << "import os" << endl;

		string cmd = "python " + share_data->instant_ngp_path + "run.py";
		//cmd += " --gui";
		cmd += " --train";
		cmd += " --n_steps " + to_string(share_data->n_steps);

		if (!nbv_test) {
			cmd += " --scene " + share_data->gt_path + "/" + trian_json_file + ".json ";
			cmd += " --test_transforms " + share_data->gt_path + "/" + test_json_file + ".json ";
			cmd += " --save_metrics " + share_data->gt_path + "/" + trian_json_file + ".txt ";
		}
		else {
			cmd += " --scene " + share_data->save_path + "/json/" + trian_json_file + ".json ";
			if (ensemble_id == -1) {
				cmd += " --test_transforms " + share_data->gt_path + "/" + test_json_file + ".json ";
				cmd += " --save_metrics " + share_data->save_path + "/metrics/" + trian_json_file + ".txt ";
			}
			else {
				cmd += " --screenshot_transforms " + share_data->save_path + "/render_json/" + trian_json_file + ".json ";
				cmd += " --screenshot_dir " + share_data->save_path + "/render/" + trian_json_file + "/ensemble_" + to_string(ensemble_id) + "/";
			}
		}

		string python_cmd = "os.system(\'" + cmd + "\')";
		fout_py << python_cmd << endl;
		fout_py.close();

		ofstream fout_py_ready(share_data->instant_ngp_path + "interact/ready_c++.txt");
		fout_py_ready.close();

		ifstream fin_over(share_data->instant_ngp_path + "interact/ready_py.txt");
		while (!fin_over.is_open()) {
			boost::this_thread::sleep(boost::posix_time::seconds(1));
			fin_over.open(share_data->instant_ngp_path + "interact/ready_py.txt");
		}
		fin_over.close();
		boost::this_thread::sleep(boost::posix_time::seconds(1));
		remove((share_data->instant_ngp_path + "interact/ready_py.txt").c_str());

		double cost_time = (clock() - now_time) / CLOCKS_PER_SEC;
		cout << "train and eval with executed time " << cost_time << " s." << endl;

		if (nbv_test) {
			if (ensemble_id == -1) {
				ofstream fout_time(share_data->save_path + "/train_time/" + trian_json_file + ".txt");
				fout_time << cost_time << endl;
				fout_time.close();
			}
		}

		return 0;
	}

	//first_view_id是测试视点空间（64）中0 0 1的id，init_view_ids是5覆盖情况下的初始视点集合
	int nbv_loop(int first_view_id = -1, vector<int> init_view_ids = vector<int>(), int test_id = 0) {

		if (init_views.size() == 0) {
			cout << "init_views is empty. read init (5) coverage view space first." << endl;
			return -1;
		}

		if (first_view_id == -1) {
			first_view_id = 0;
			cout << "first_view_id is -1. use 0 as id." << endl;
		}

		if (init_view_ids.size() == 0) {
			init_view_ids.push_back(1);
			cout << "init_view_ids is empty. use 5 coverage view space top id." << endl;
		}

		if (share_data->method_of_IG != PVBCoverage) {
			cout << "method_of_IG is not PVBCoverage. Read view budget." << endl;
			ifstream fin_view_budget(share_data->pre_path + "Compare/ShapeNet/" + share_data->name_of_pcd +  "_m4_v" + to_string(init_view_ids.size()) + "_t" + to_string(test_id) + "/view_budget.txt");
			if (fin_view_budget.is_open()) {
				int view_budget;
				fin_view_budget >> view_budget;
				fin_view_budget.close();
				share_data->num_of_max_iteration = view_budget - 1;
				cout << "readed view_budget is " << view_budget << endl;
			}
			else {
				cout << "view_budget.txt is not exist. use deaulft as view budget." << endl;
			}
			cout << "num_of_max_iteration is set as " << share_data->num_of_max_iteration << endl;
		}

		share_data->save_path += "_v" + to_string(init_view_ids.size());
		share_data->save_path += "_t" + to_string(test_id);
		share_data->access_directory(share_data->save_path + "/json");
		share_data->access_directory(share_data->save_path + "/render_json");
		share_data->access_directory(share_data->save_path + "/metrics");
		share_data->access_directory(share_data->save_path + "/render");
		share_data->access_directory(share_data->save_path + "/train_time");
		share_data->access_directory(share_data->save_path + "/infer_time");
		share_data->access_directory(share_data->save_path + "/movement");

		ifstream check_fininshed(share_data->save_path + "/run_time.txt");
		if (check_fininshed.is_open()) {
			double run_time;
			check_fininshed >> run_time;
			check_fininshed.close();
			if (run_time >= 0) {
				cout << "run_time.txt is exist. nbv_loop is finished." << endl;
				return 0;
			}
		}

		//json root
		Json::Value root_nbvs;
		root_nbvs["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root_nbvs["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root_nbvs["fl_x"] = share_data->color_intrinsics.fx;
		root_nbvs["fl_y"] = share_data->color_intrinsics.fy;
		root_nbvs["k1"] = share_data->color_intrinsics.coeffs[0];
		root_nbvs["k2"] = share_data->color_intrinsics.coeffs[1];
		root_nbvs["k3"] = share_data->color_intrinsics.coeffs[2];
		root_nbvs["p1"] = share_data->color_intrinsics.coeffs[3];
		root_nbvs["p2"] = share_data->color_intrinsics.coeffs[4];
		root_nbvs["cx"] = share_data->color_intrinsics.ppx;
		root_nbvs["cy"] = share_data->color_intrinsics.ppy;
		root_nbvs["w"] = share_data->color_intrinsics.width;
		root_nbvs["h"] = share_data->color_intrinsics.height;
		root_nbvs["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root_nbvs["scale"] = 0.5 / share_data->predicted_size;
		root_nbvs["offset"][0] = 0.5 + share_data->object_center_world(2);
		root_nbvs["offset"][1] = 0.5 + share_data->object_center_world(0);
		root_nbvs["offset"][2] = 0.5 + share_data->object_center_world(1);

		Json::Value root_render;
		root_render["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root_render["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root_render["fl_x"] = share_data->color_intrinsics.fx / 16.0;
		root_render["fl_y"] = share_data->color_intrinsics.fy / 16.0;
		root_render["k1"] = 0;
		root_render["k2"] = 0;
		root_render["k3"] = 0;
		root_render["p1"] = 0;
		root_render["p2"] = 0;
		root_render["cx"] = share_data->color_intrinsics.ppx / 16.0;
		root_render["cy"] = share_data->color_intrinsics.ppy / 16.0;
		root_render["w"] = share_data->color_intrinsics.width / 16.0;
		root_render["h"] = share_data->color_intrinsics.height / 16.0;
		root_render["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root_render["scale"] = 0.5 / share_data->predicted_size;
		root_render["offset"][0] = 0.5 + share_data->object_center_world(2);
		root_render["offset"][1] = 0.5 + share_data->object_center_world(0);
		root_render["offset"][2] = 0.5 + share_data->object_center_world(1);

		//5覆盖初始化中除了0 0 1视点外的视点
		int first_init_view_id = -1;
		for (int i = 0; i < init_view_ids.size(); i++) {
			if (fabs(init_views[init_view_ids[i]].init_pos(0)) < 1e-6 && fabs(init_views[init_view_ids[i]].init_pos(1)) < 1e-6 && fabs(init_views[init_view_ids[i]].init_pos(2) - share_data->view_space_radius) < 1e-6) {
				first_init_view_id = init_view_ids[i];
				continue;
			}
			//get json
			Json::Value view_image;
			view_image["file_path"] = "../../../../Coverage_images/ShapeNet/" + share_data->name_of_pcd + "/" + to_string(init_views.size()) + "/rgbaClip_" + to_string(init_view_ids[i]) + ".png";
			Json::Value transform_matrix;
			for (int k = 0; k < 4; k++) {
				Json::Value row;
				for (int l = 0; l < 4; l++) {
					init_views[init_view_ids[i]].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = share_data->now_camera_pose_world * init_views[init_view_ids[i]].pose.inverse();
					//x,y,z->y,z,x
					Eigen::Matrix4d pose;
					pose << 0, 0, 1, 0,
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;
					//x,y,z->x,-y,-z
					Eigen::Matrix4d pose_1;
					pose_1 << 1, 0, 0, 0,
						0, -1, 0, 0,
						0, 0, -1, 0,
						0, 0, 0, 1;
					view_pose_world = pose * view_pose_world * pose_1;
					row.append(view_pose_world(k, l));
				}
				transform_matrix.append(row);
			}
			view_image["transform_matrix"] = transform_matrix;
			root_nbvs["frames"].append(view_image);
		}
		double init_dis = 0.0;
		if(init_view_ids.size() > 1) {
			Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, init_views, init_view_ids, first_init_view_id);
			init_dis = gloabl_path_planner->solve();
			init_view_ids = gloabl_path_planner->get_path_id_set();
			//反转路径
			reverse(init_view_ids.begin(), init_view_ids.end());
			cout << "init_dis: " << init_dis << endl;
			delete gloabl_path_planner;
		}
		//输出起始路径
		ofstream fout_init_path(share_data->save_path + "/movement/init_path.txt");
		for (int i = 0; i < init_view_ids.size(); i++) {
			fout_init_path << init_view_ids[i] << endl;
		}
		fout_init_path.close();

		//double total_movement_cost = init_dis;
		double total_movement_cost = 0.0;
		ofstream fout_move_first(share_data->save_path + "/movement/" + to_string(-1) + ".txt");
		fout_move_first << first_view_id << '\t' << init_dis << '\t' << total_movement_cost << endl;
		fout_move_first.close();

		//初始视点
		vector<int> chosen_nbvs;
		chosen_nbvs.push_back(first_view_id);
		set<int> chosen_nbvs_set;
		chosen_nbvs_set.insert(first_view_id);

		vector<int> oneshot_views;

		//循环NBV
		double now_time = clock();
		int iteration = 0;
		while (true) {
			//生成当前视点集合json
			Json::Value now_nbvs_json(root_nbvs);
			Json::Value now_render_json(root_render);
			for (int i = 0; i < share_data->num_of_views; i++) {
				Json::Value view_image;
				view_image["file_path"] = "../../../../Coverage_images/ShapeNet/" + share_data->name_of_pcd + "/" + to_string(share_data->num_of_views) + "/rgbaClip_" + to_string(i) + ".png";
				Json::Value transform_matrix;
				for (int k = 0; k < 4; k++) {
					Json::Value row;
					for (int l = 0; l < 4; l++) {
						view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
						Eigen::Matrix4d view_pose_world = share_data->now_camera_pose_world * view_space->views[i].pose.inverse();
						//x,y,z->y,z,x
						Eigen::Matrix4d pose;
						pose << 0, 0, 1, 0,
							1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 0, 1;
						//x,y,z->x,-y,-z
						Eigen::Matrix4d pose_1;
						pose_1 << 1, 0, 0, 0,
							0, -1, 0, 0,
							0, 0, -1, 0,
							0, 0, 0, 1;
						view_pose_world = pose * view_pose_world * pose_1;
						row.append(view_pose_world(k, l));
					}
					transform_matrix.append(row);
				}
				view_image["transform_matrix"] = transform_matrix;
				if (chosen_nbvs_set.count(i)) now_nbvs_json["frames"].append(view_image);
				else now_render_json["frames"].append(view_image);
			}
			Json::StyledWriter writer_nbvs_json;
			ofstream fout_nbvs_json(share_data->save_path + "/json/" + to_string(iteration) + ".json");
			fout_nbvs_json << writer_nbvs_json.write(now_nbvs_json);
			fout_nbvs_json.close();
			Json::StyledWriter writer_render_json;
			ofstream fout_render_json(share_data->save_path + "/render_json/" + to_string(iteration) + ".json");
			fout_render_json << writer_render_json.write(now_render_json);
			fout_render_json.close();

			//如果需要测试，则训练当前视点集合
			cout << "iteration " << iteration << endl;
			cout<< "chosen_nbvs: ";
			for (int i = 0; i < chosen_nbvs.size(); i++) {
				cout << chosen_nbvs[i] << ' ';
			}
			cout << endl;
			//if (share_data->evaluate) {
			//	cout << "evaluating..." << endl;
			//	train_by_instantNGP(to_string(iteration), "100", true);
			//	ifstream fin_metrics(share_data->save_path + "/metrics/" + to_string(iteration) + ".txt");
			//	string metric_name;
			//	double metric_value;
			//	while (fin_metrics >> metric_name >> metric_value) {
			//		cout << metric_name << ": " << metric_value << endl;
			//	}
			//	fin_metrics.close();
			//}

			//如果达到最大迭代次数，则结束
			if (iteration == share_data->num_of_max_iteration) {
				//保存运行时间
				double loops_time = (clock() - now_time) / CLOCKS_PER_SEC;
				ofstream fout_loops_time(share_data->save_path + "/run_time.txt");
				fout_loops_time << loops_time << endl;
				fout_loops_time.close();
				cout << "run time " << loops_time << " ms." << endl;
				////如果不需要逐步测试，则训练最终视点集合
				if (share_data->evaluate) {
					cout << "final evaluating..." << endl;
					train_by_instantNGP(to_string(iteration), "100", true);
					ifstream fin_metrics(share_data->save_path + "/metrics/" + to_string(iteration) + ".txt");
					string metric_name;
					double metric_value;
					while (fin_metrics >> metric_name >> metric_value) {
						cout << metric_name << ": " << metric_value << endl;
					}
					fin_metrics.close();
				}
				break;
			}

			//根据不同方法获取NBV
			double infer_time = clock();
			int next_view_id = -1;
			double largest_view_uncertainty = -1e100;
			int best_view_id = -1;
			switch (share_data->method_of_IG) {
				case 0: //RandomIterative
					next_view_id = rand() % share_data->num_of_views;
					while (chosen_nbvs_set.count(next_view_id)) {
						next_view_id = rand() % share_data->num_of_views;
					}
					break;

				case 1: //RandomOneshot
					if (oneshot_views.size() == 0) {
						//随机50次，选出分布最均匀的一个，即相互之间距离之和最大
						int check_num = 50;
						set<int> best_oneshot_views_set;
						double largest_pair_dis = -1e100;
						while (check_num--) {
							set<int> oneshot_views_set;
							oneshot_views_set.insert(first_view_id);
							for (int i = 0; i < share_data->num_of_max_iteration; i++) {
								int random_view_id = rand() % share_data->num_of_views;
								while (oneshot_views_set.count(random_view_id)) {
									random_view_id = rand() % share_data->num_of_views;
								}
								oneshot_views_set.insert(random_view_id);
							}
							double now_dis = 0;
							for (auto it = oneshot_views_set.begin();it != oneshot_views_set.end(); it++) {
								auto it2 = it;
								it2++;
								while (it2 != oneshot_views_set.end()) {
									now_dis += (view_space->views[*it].init_pos - view_space->views[*it2].init_pos).norm();
									it2++;
								}
							}
							if (now_dis > largest_pair_dis) {
								largest_pair_dis = now_dis;
								best_oneshot_views_set = oneshot_views_set;
								cout<< "largest_pair_dis: " << largest_pair_dis << endl;
							}
						}
						set<int> oneshot_views_set = best_oneshot_views_set;

						for (auto it = oneshot_views_set.begin(); it != oneshot_views_set.end(); it++) {
							oneshot_views.push_back(*it);
						}
						cout << "oneshot_views num is: " << oneshot_views.size() << endl;
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, view_space->views, oneshot_views, first_view_id);
						double total_dis = gloabl_path_planner->solve();
						oneshot_views = gloabl_path_planner->get_path_id_set();
						if (oneshot_views.size() != share_data->num_of_max_iteration + 1) {
							cout << "oneshot_views.size() != share_data->num_of_max_iteration + 1" << endl;
						}
						cout<< "total_dis: " << total_dis << endl;
						delete gloabl_path_planner;
						//删除初始视点
						oneshot_views.erase(oneshot_views.begin());
						//更新迭代次数，取出NBV
						share_data->num_of_max_iteration = oneshot_views.size();
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					else {
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					break;

				case 2: //EnsembleRGB
					//交给instantngp训练
					for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
						train_by_instantNGP(to_string(iteration), "100", true, ensemble_id);
					}
					//计算评价指标
					for (int i = 0; i < share_data->num_of_views; i++) {
						if (chosen_nbvs_set.count(i)) continue;
						vector<cv::Mat> rgb_images;
						for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
							cv::Mat rgb_image = cv::imread(share_data->save_path + "/render/" + to_string(iteration) + "/ensemble_" + to_string(ensemble_id) + "/rgbaClip_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
							rgb_images.push_back(rgb_image);
						}
						//使用ensemble计算uncertainty
						double view_uncertainty = 0.0;
						for (int j = 0; j < rgb_images[0].rows; j++) {
							for (int k = 0; k < rgb_images[0].cols; k++) {
								double mean_r = 0.0;
								double mean_g = 0.0;
								double mean_b = 0.0;
								for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
									cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
									//注意不要归一化，不然会导致取log为负
									mean_r += rgba[0];
									mean_g += rgba[1];
									mean_b += rgba[2];
								}
								//计算方差
								mean_r /= share_data->ensemble_num;
								mean_g /= share_data->ensemble_num;
								mean_b /= share_data->ensemble_num;
								double variance_r = 0.0;
								double variance_g = 0.0;
								double variance_b = 0.0;
								for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
									cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
									variance_r += (rgba[0] - mean_r) * (rgba[0] - mean_r);
									variance_g += (rgba[1] - mean_g) * (rgba[1] - mean_g);
									variance_b += (rgba[2] - mean_b) * (rgba[2] - mean_b);
								};
								variance_r /= share_data->ensemble_num;
								variance_g /= share_data->ensemble_num;
								variance_b /= share_data->ensemble_num;
								if (variance_r > 1e-10) view_uncertainty += log(variance_r);
								if (variance_g > 1e-10) view_uncertainty += log(variance_g);
								if (variance_b > 1e-10) view_uncertainty += log(variance_b);
							}
						}
						//cout << i << " " << view_uncertainty << " " << largest_view_uncertainty << endl;
						if (view_uncertainty > largest_view_uncertainty) {
							largest_view_uncertainty = view_uncertainty;
							best_view_id = i;
						}
						rgb_images.clear();
						rgb_images.shrink_to_fit();
					}
					//选择最好的视点
					next_view_id = best_view_id;
					break;
				
				case 3: //EnsembleRGBDensity	
					//交给instantngp训练
					for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
						train_by_instantNGP(to_string(iteration), "100", true, ensemble_id);
					}
					//计算评价指标
					for (int i = 0; i < share_data->num_of_views; i++) {
						if (chosen_nbvs_set.count(i)) continue;
						vector<cv::Mat> rgb_images;
						for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
							cv::Mat rgb_image = cv::imread(share_data->save_path + "/render/" + to_string(iteration) + "/ensemble_" + to_string(ensemble_id) + "/rgbaClip_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
							rgb_images.push_back(rgb_image);
						}
						//使用ensemble计算uncertainty，其中density存于alpha通道
						double view_uncertainty = 0.0;
						for (int j = 0; j < rgb_images[0].rows; j++) {
							for (int k = 0; k < rgb_images[0].cols; k++) {
								double mean_r = 0.0;
								double mean_g = 0.0;
								double mean_b = 0.0;
								double mean_density = 0.0;
								for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
									cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
									//注意不要归一化，不然会导致取log为负
									mean_r += rgba[0];
									mean_g += rgba[1];
									mean_b += rgba[2];
									//注意alpha通道存储的是density，要归一化到0-1
									mean_density += rgba[3] / 255.0;
								}
								mean_r /= share_data->ensemble_num;
								mean_g /= share_data->ensemble_num;
								mean_b /= share_data->ensemble_num;
								mean_density /= share_data->ensemble_num;
								//cout << mean_r << " " << mean_g << " " << mean_b << " " << mean_density << endl;
								//计算方差
								double variance_r = 0.0;
								double variance_g = 0.0;
								double variance_b = 0.0;
								for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
									cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
									variance_r += (rgba[0] - mean_r) * (rgba[0] - mean_r);
									variance_g += (rgba[1] - mean_g) * (rgba[1] - mean_g);
									variance_b += (rgba[2] - mean_b) * (rgba[2] - mean_b);
								};
								variance_r /= share_data->ensemble_num;
								variance_g /= share_data->ensemble_num;
								variance_b /= share_data->ensemble_num;
								view_uncertainty += (variance_r + variance_g + variance_b) / 3.0;
								view_uncertainty += (1.0 - mean_density) * (1.0 - mean_density);
							}
						}
						//cout << i << " " << view_uncertainty << " " << largest_view_uncertainty << endl;
						if (view_uncertainty > largest_view_uncertainty) {
							largest_view_uncertainty = view_uncertainty;
							best_view_id = i;
						}
						rgb_images.clear();
						rgb_images.shrink_to_fit();
					}
					//选择最好的视点
					next_view_id = best_view_id;
					break;

				case 4: //PVBCoverage
					if (oneshot_views.size() == 0) {
						////通过网络获取视点预算
						share_data->access_directory(share_data->pvb_path + "data/images");
						for (int i = 0; i < init_view_ids.size(); i++) {
							ofstream fout_image(share_data->pvb_path + "data/images/" + to_string(init_view_ids[i]) + ".png", std::ios::binary);
							ifstream fin_image(share_data->gt_path + "/" + to_string(init_views.size()) + "/rgbaClip_" + to_string(init_view_ids[i]) + ".png", std::ios::binary);
							fout_image << fin_image.rdbuf();
							fout_image.close();
							fin_image.close();
						}
						ofstream fout_ready(share_data->pvb_path + "data/ready_c++.txt");
						fout_ready.close();
						//等待python程序结束
						ifstream fin_over(share_data->pvb_path + "data/ready_py.txt");
						while (!fin_over.is_open()) {
							boost::this_thread::sleep(boost::posix_time::milliseconds(100));
							fin_over.open(share_data->pvb_path + "data/ready_py.txt");
						}
						fin_over.close();
						boost::this_thread::sleep(boost::posix_time::milliseconds(100));
						remove((share_data->pvb_path + "data/ready_py.txt").c_str());
						////读取view budget, bus25 gt20, airplane0 gt14
						int view_budget = -1;
						ifstream fin_view_budget(share_data->pvb_path + "data/view_budget.txt");
						if (!fin_view_budget.is_open()) {
							cout << "view budget file not found!" << endl;
						}
						fin_view_budget >> view_budget;
						fin_view_budget.close();
						cout << "view budget is: " << view_budget << endl;
						//读取coverage view space
						share_data->num_of_views = view_budget;
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "coverage view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						view_space.reset();
						view_space = make_shared<View_Space>(share_data);
						int now_first_view_id = -1;
						for (int i = 0; i < share_data->num_of_views; i++) {
							if (fabs(view_space->views[i].init_pos(0)) < 1e-6 && fabs(view_space->views[i].init_pos(1)) < 1e-6 && fabs(view_space->views[i].init_pos(2) - share_data->view_space_radius) < 1e-6) {
								now_first_view_id = i;
							}
							oneshot_views.push_back(i);
						}
						if (now_first_view_id == -1) {
							cout << "can not find now view id" << endl;
						}
						chosen_nbvs.clear();
						chosen_nbvs.push_back(now_first_view_id);
						chosen_nbvs_set.clear();
						chosen_nbvs_set.insert(now_first_view_id);
						//执行全局路径规划
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, view_space->views, oneshot_views, now_first_view_id);
						double total_dis = gloabl_path_planner->solve();
						oneshot_views = gloabl_path_planner->get_path_id_set();
						cout << "total_dis: " << total_dis << endl;
						delete gloabl_path_planner;
						//保存所有视点个数
						ofstream fout_iteration(share_data->save_path + "/view_budget.txt");
						fout_iteration << oneshot_views.size() << endl;
						//删除初始视点
						oneshot_views.erase(oneshot_views.begin());
						//更新迭代次数，取出NBV
						share_data->num_of_max_iteration = oneshot_views.size();
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					else {
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					break;
			}
			chosen_nbvs.push_back(next_view_id);
			chosen_nbvs_set.insert(next_view_id);
			cout << "next_view_id: " << next_view_id << endl;

			infer_time = (clock() - infer_time) / CLOCKS_PER_SEC;
			ofstream fout_infer_time(share_data->save_path + "/infer_time/" + to_string(iteration) + ".txt");
			fout_infer_time << infer_time << endl;
			fout_infer_time.close();

			//运动代价：视点id，当前代价，总体代价
			int now_nbv_id = chosen_nbvs[iteration];
			int next_nbv_id = chosen_nbvs[iteration + 1];
			pair<int, double> local_path = get_local_path(view_space->views[now_nbv_id].init_pos.eval(), view_space->views[next_nbv_id].init_pos.eval(), (share_data->object_center_world + Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), share_data->predicted_size);
			total_movement_cost += local_path.second;
			cout << "local path: " << local_path.second << " total: " << total_movement_cost << endl;

			ofstream fout_move(share_data->save_path + "/movement/" + to_string(iteration) + ".txt");
			fout_move << next_nbv_id << '\t' << local_path.second << '\t' << total_movement_cost << endl;
			fout_move.close();

			//更新迭代次数
			iteration++;
		}

		chosen_nbvs.clear();
		chosen_nbvs.shrink_to_fit();
		chosen_nbvs_set.clear();
		oneshot_views.clear();
		oneshot_views.shrink_to_fit();

		return 0;
	}

};

#define ViewCover 0
#define ViewNovel 1
#define GetSizeTest 2
#define GetCoverage 3
#define InstantNGP 4
#define ReadLabel 5
#define GetDataset 6
#define TestObjects 7
#define ShapeNetPreProcess 10
#define GetCleanData 11
#define GetPathPlan 20
#define ViewPlanning 21

int main()
{
	//Init
	srand(time(0));
	ios::sync_with_stdio(false);
	int mode;
	cout << "input mode:";
	cin >> mode;
	//测试集
	vector<string> names;
	cout << "input models:" << endl;
	string name;
	while (cin >> name) {
		if (name == "-1") break;
		names.push_back(name);
	}
	//选取模式
	if (mode == ViewCover) {
		shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", "", -1);
		shared_ptr<NBV_Net_Labeler> labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
		labeler->get_cover_view_cases();
		cout << "global labeler use_count is " << labeler.use_count() << endl;
		cout << "global share_data use_count is " << share_data.use_count() << endl;
		labeler.reset();
		share_data.reset();
	}
	else if (mode == ViewNovel) {
		shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", "", -1);
		shared_ptr<NBV_Net_Labeler> labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
		labeler->get_novel_view_cases();
		cout << "global labeler use_count is " << labeler.use_count() << endl;
		cout << "global share_data use_count is " << share_data.use_count() << endl;
		labeler.reset();
		share_data.reset();
	}
	else if (mode == GetSizeTest) {
		for (int i = 0; i < names.size(); i++) {
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1);
			ifstream size_reader(share_data->gt_path + "/size.txt");
			if (!size_reader.is_open()) {
				shared_ptr<NBV_Net_Labeler> labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
				//labeler->get_train_test_novel();
				cout << "global labeler use_count is " << labeler.use_count() << endl;
				labeler.reset();
			}
			cout << "global share_data use_count is " << share_data.use_count() << endl;
			share_data.reset();
		}
	}
	else if (mode == GetCoverage){
		for (int i = 0; i < names.size(); i++) {
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1);
			shared_ptr<NBV_Net_Labeler> labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
			if (labeler->object_is_ok_size) {
				////3-90_+1; 3-50_+2
				for (int num_of_coverage_views = 3; num_of_coverage_views <= share_data->coverage_view_num_max; num_of_coverage_views += share_data->coverage_view_num_add) {
				//for (int num_of_coverage_views = 5; num_of_coverage_views <= 60; num_of_coverage_views += 1) {
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						labeler->view_space.reset();
						labeler->view_space = make_shared<View_Space>(share_data);
						//get images
						labeler->get_coverage();
					}
				}
				//100
				{
					int num_of_coverage_views = 100;
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						labeler->view_space.reset();
						labeler->view_space = make_shared<View_Space>(share_data);
						//get images
						labeler->get_coverage();
					}
				}
				////540/144
				//{
				//	int num_of_coverage_views = 540;
				//	ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
				//	if (!fin_json.is_open()) {
				//		share_data->num_of_views = num_of_coverage_views;
				//		//read viewspace again
				//		ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
				//		share_data->pt_sphere.clear();
				//		share_data->pt_sphere.resize(share_data->num_of_views);
				//		for (int i = 0; i < share_data->num_of_views; i++) {
				//			share_data->pt_sphere[i].resize(3);
				//			for (int j = 0; j < 3; j++) {
				//				fin_sphere >> share_data->pt_sphere[i][j];
				//			}
				//		}
				//		cout << "view space size is: " << share_data->pt_sphere.size() << endl;
				//		Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
				//		share_data->pt_norm = pt0.norm();
				//		//reget viewspace
				//		labeler->view_space.reset();
				//		labeler->view_space = make_shared<View_Space>(share_data);
				//		//get images
				//		labeler->get_coverage();
				//	}
				//}
				////32/34/35
				//vector<int> test_Statistic;
				//test_Statistic.push_back(32);
				//test_Statistic.push_back(34);
				////test_Statistic.push_back(35);
				//for (int j = 0; j < test_Statistic.size(); j++){
				//	int num_of_coverage_views = test_Statistic[j];
				//	ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
				//	if (!fin_json.is_open()) {
				//		share_data->num_of_views = num_of_coverage_views;
				//		//read viewspace again
				//		ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
				//		share_data->pt_sphere.clear();
				//		share_data->pt_sphere.resize(share_data->num_of_views);
				//		for (int i = 0; i < share_data->num_of_views; i++) {
				//			share_data->pt_sphere[i].resize(3);
				//			for (int j = 0; j < 3; j++) {
				//				fin_sphere >> share_data->pt_sphere[i][j];
				//			}
				//		}
				//		cout << "view space size is: " << share_data->pt_sphere.size() << endl;
				//		Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
				//		share_data->pt_norm = pt0.norm();
				//		//reget viewspace
				//		labeler->view_space.reset();
				//		labeler->view_space = make_shared<View_Space>(share_data);
				//		//get images
				//		labeler->get_coverage();
				//	}
				//}
			}
			cout << "global labeler use_count is " << labeler.use_count() << endl;
			cout << "global share_data use_count is " << share_data.use_count() << endl;
			labeler.reset();
			share_data.reset();
		}
	}
	else if (mode == InstantNGP) {
		for (int i = 0; i < names.size(); i++) {
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1);
			shared_ptr<NBV_Net_Labeler> labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
			if (labeler->object_is_ok_size) {
				cout << "coverage_view_num_max is " << share_data->coverage_view_num_max << endl;
				cout << "coverage_view_num_add is " << share_data->coverage_view_num_add << endl;
				//3-100_+1; 3-50_+2
				for (int num_of_coverage_views = 3; num_of_coverage_views <= share_data->coverage_view_num_max; num_of_coverage_views += share_data->coverage_view_num_add) {
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".txt");
					if (!fin_json.is_open()) {
						//train
						labeler->train_by_instantNGP(to_string(num_of_coverage_views));
					}
				}
				ifstream fin_json_trainviews(share_data->gt_path + "/100.txt");
				if (!fin_json_trainviews.is_open()) {
					//train
					labeler->train_by_instantNGP("100");
				}
			}
			cout << "global labeler use_count is " << labeler.use_count() << endl;
			cout << "global share_data use_count is " << share_data.use_count() << endl;
			labeler.reset();
			share_data.reset();
		}
	}
	else if (mode == ReadLabel) {
		vector<int> converges;
		vector<vector<double>> curves;
		vector<vector<long long>> label_gaps, label_gradients;
		for (int i = 0; i < names.size(); i++) {
			//int id_of_batch = 0;
			int id_of_batch = i / 3000;
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, id_of_batch);
			ifstream fin_label(share_data->pre_path + "Coverage_images/ShapeNet_" + to_string(id_of_batch) + "_label" + "/" + names[i] + "/label.txt");
			if (!fin_label.is_open()) {
				cout << "Check! label missing " << names[i] << endl;
				break;
			}
			int curve_converged;
			vector<double> curve;
			vector<long long> label_gap, label_gradient;
			string temp_string;
			long long temp_int;
			double temp_double;
			//read label
			fin_label >> temp_string >> curve_converged;
			if (temp_string != "Converged") {
				cout << "label wrong " << temp_string << " " << curve_converged << endl;
			}
			converges.push_back(curve_converged);
			for (int num_of_coverage_views = 3; num_of_coverage_views <= 100; num_of_coverage_views ++) {
				fin_label >> temp_int >> temp_double;
				if (temp_int != num_of_coverage_views) {
					cout << "label wrong " << temp_int << endl;
				}
				curve.push_back(temp_double);
			}
			curves.push_back(curve);
			for (int gap_value = 0; gap_value <= 10; gap_value++) {
				fin_label >> temp_string;
				if (temp_string != "gap") {
					cout << "label wrong " << temp_string << endl;
				}
				fin_label >> temp_string;
				fin_label >> temp_int;
				label_gap.push_back(temp_int);
			}
			label_gaps.push_back(label_gap);
			for (int gradient_value = 0; gradient_value < 20; gradient_value++) {
				fin_label >> temp_string;
				if (temp_string != "gradient") {
					cout << "label wrong " << temp_string << endl;
				}
				fin_label >> temp_string;
				fin_label >> temp_int;
				label_gradient.push_back(temp_int);
			}
			label_gradients.push_back(label_gradient);
			fin_label.close();
			cout << "global share_data use_count is " << share_data.use_count() << endl;
			share_data.reset();
		}
		//show converges
		for (int i = 0; i < names.size(); i++) {
			if (!converges[i]) {
				cout << "no converged " << names[i] << endl;
			}
		}			
		//calculate mean and std
		ofstream fout_label("D:/Data/NeRF_coverage/label_mean_std.txt");
		ofstream fout_distribution("D:/Data/NeRF_coverage/label_distribution.txt");
		vector<double> mean_label_gap, std_label_gap, mean_label_gradient, std_label_gradient;
		fout_label << "type" << "\t" << "value" << "\t" << "mean" << "\t" << "std" << "\t" << "fail_num" << "\t" << "min" << "\t" << "max" << "\n";
		for (int gap_value = 0; gap_value <= 10; gap_value++) {
			fout_label << "gap\t" << gap_value <<"%";
			double sum = 0;
			int count = 0;
			double max = -1;
			double min = 100000000;
			vector<int> distribution;
			distribution.resize(101);
			for (int i = 0; i < names.size(); i++) {
				if (converges[i] && label_gaps[i][gap_value] != -1) {
					sum += label_gaps[i][gap_value];
					if (label_gaps[i][gap_value] > max) {
						max = label_gaps[i][gap_value];
					}
					if (label_gaps[i][gap_value] < min) {
						min = label_gaps[i][gap_value];
					}
					distribution[label_gaps[i][gap_value]]++;
					count++;
				}
			}
			mean_label_gap.push_back(sum / count);
			fout_label << "\t" << sum / count;
			sum = 0;
			for (int i = 0; i < names.size(); i++) {
				if (converges[i] && label_gaps[i][gap_value] != -1) {
					sum += (label_gaps[i][gap_value] - mean_label_gap[gap_value]) * (label_gaps[i][gap_value] - mean_label_gap[gap_value]);
				}
			}
			std_label_gap.push_back(sqrt(sum / (count - 1)));
			fout_label << "\t" << sqrt(sum / (count - 1));
			fout_label << "\t" << label_gaps.size() - (count - 1);
			fout_label << "\t" << min << "\t" << max << "\n";
			fout_distribution << "gap\t" << gap_value << "%\n";
			for (int i = min; i <= max; i++) {
				fout_distribution << i << "\t" << distribution[i] << "\n";
			}
			fout_distribution << "\n";
		}
		for (int gradient_value = 0; gradient_value < 20; gradient_value++) {
			fout_label << "gradient\t" << to_string(0.01 * (gradient_value + 1));
			double sum = 0;
			int count = 0;
			double max = -1;
			double min = 100000000;
			vector<int> distribution;
			distribution.resize(101);
			for (int i = 0; i < names.size(); i++) {
				if (converges[i] && label_gradients[i][gradient_value] != -1) {
					sum += label_gradients[i][gradient_value];
					if (label_gradients[i][gradient_value] > max) {
						max = label_gradients[i][gradient_value];
					}
					if (label_gradients[i][gradient_value] < min) {
						min = label_gradients[i][gradient_value];					
					}
					distribution[label_gradients[i][gradient_value]]++;
					count++;
				}
			}
			mean_label_gradient.push_back(sum / count);
			fout_label << "\t" << sum / count;
			sum = 0;
			for (int i = 0; i < names.size(); i++) {
				if (converges[i] && label_gradients[i][gradient_value] != -1) {
					sum += (label_gradients[i][gradient_value] - mean_label_gradient[gradient_value]) * (label_gradients[i][gradient_value] - mean_label_gradient[gradient_value]);
				}
			}
			std_label_gradient.push_back(sqrt(sum / (count - 1)));
			fout_label << "\t" << sqrt(sum / (count - 1));
			fout_label << "\t" << label_gradients.size() - (count - 1);
			fout_label << "\t" << min << "\t" << max << "\n";
			fout_distribution << "gradient\t" << to_string(0.01 * (gradient_value + 1)) << "\n";
			for (int i = min; i <= max; i++) {
				fout_distribution << i << "\t" << distribution[i] << "\n";
			}
			fout_distribution << "\n";
		}
		fout_label.close();
		fout_distribution.close();
	}
	else if (mode == GetDataset) {
		//gap 5% with drop labels larger than 64
		string label_str = "gradient";
		int label_value_index = 1;
		//3 Sigma Law to drop outliers
		int min_coverage_views = 13;
		int max_coverage_views = 58;

		vector<int> converges;
		vector<int> labels;
		for (int i = 0; i < names.size(); i++) {
			int id_of_batch = i / 3000;
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, id_of_batch);
			ifstream fin_label(share_data->pre_path + "Coverage_images/ShapeNet_" + to_string(id_of_batch) + "_label" + "/" + names[i] + "/label.txt");
			if (!fin_label.is_open()) {
				cout << "Check! label missing " << names[i] << endl;
				break;
			}
			int curve_converged;
			string temp_string;
			int temp_int;
			double temp_double;
			//read label
			fin_label >> temp_string >> curve_converged;
			if (temp_string != "Converged") {
				cout << "label wrong " << temp_string << " " << curve_converged << endl;
			}
			converges.push_back(curve_converged);
			for (int num_of_coverage_views = 3; num_of_coverage_views <= 100; num_of_coverage_views++) {
				fin_label >> temp_int >> temp_double;
				if (temp_int != num_of_coverage_views) {
					cout << "label wrong " << temp_int << endl;
				}
			}
			for (int gap_value = 0; gap_value <= 10; gap_value++) {
				fin_label >> temp_string;
				if (temp_string != "gap") {
					cout << "label wrong " << temp_string << endl;
				}
				fin_label >> temp_string;
				fin_label >> temp_int;
				if (label_str == "gap" && label_value_index == gap_value) {
					labels.push_back(temp_int);
				}
			}
			for (int gradient_value = 0; gradient_value < 20; gradient_value++) {
				fin_label >> temp_string;
				if (temp_string != "gradient") {
					cout << "label wrong " << temp_string << endl;
				}
				fin_label >> temp_string;
				fin_label >> temp_int;
				if (label_str == "gradient" && label_value_index == gradient_value) {
					labels.push_back(temp_int);
				}
			}
			fin_label.close();
			share_data.reset();
		}
		//show converges
		set<pair<int, string>> label_name_set;
		map<string,vector<int>> name_num_map;
		vector<int> label_num;
		label_num.resize(100);
		for (int i = min_coverage_views; i <= max_coverage_views; i++) {
			label_num[i] = 0;
		}
		name_num_map["tab"] = label_num;
		name_num_map["car"] = label_num;
		name_num_map["cha"] = label_num;
		name_num_map["air"] = label_num;
		name_num_map["sof"] = label_num;
		name_num_map["rif"] = label_num;
		name_num_map["lam"] = label_num;
		name_num_map["wat"] = label_num;
		name_num_map["ben"] = label_num;
		name_num_map["lou"] = label_num;
		name_num_map["cab"] = label_num;
		name_num_map["dis"] = label_num;
		name_num_map["tel"] = label_num;
		name_num_map["bus"] = label_num;
		name_num_map["bat"] = label_num;
		name_num_map["gui"] = label_num;
		name_num_map["fau"] = label_num;
		name_num_map["clo"] = label_num;
		name_num_map["flo"] = label_num;
		name_num_map["jar"] = label_num;
		double mean_label = 0, max_label = -1, min_label = 100000000;
		for (int i = 0; i < names.size(); i++) {
			if (!converges[i]) {
				cout << "no converged " << names[i] << endl;
				continue;
			}
			if (labels[i] == -1) {
				cout << "no label " << names[i] << endl;
				continue;
			}
			if (labels[i] < min_coverage_views) {
				cout << "coverage views too small " << names[i] << endl;
				continue;
			}
			if (labels[i] > max_coverage_views) {
				cout << "coverage views too large " << names[i] << endl;
				continue;
			}
			label_name_set.insert(make_pair(labels[i], names[i]));
			mean_label += labels[i];
			if (labels[i] > max_label) {
				max_label = labels[i];
			}
			if (labels[i] < min_label) {
				min_label = labels[i];
			}
			// 匹配names[i]的前三个字符
			string name_str = names[i].substr(0, 3);
			if (name_num_map.find(name_str) == name_num_map.end()) {
				cout << "name wrong " << name_str << endl;
			}
			else {
				name_num_map[name_str][labels[i]]++;
			}
			
			//生成监督对
			int id_of_batch = i / 3000;
			//shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, id_of_batch);
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, -1);
			shared_ptr<NBV_Net_Labeler> labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
			share_data->access_directory("D:/Data/NeRF_coverage/pvb_dataset/" + names[i]);
			////保存点云share_data->cloud_pcd中xyz
			//ofstream fout_xyz("D:/Data/NeRF_coverage/pvb_dataset/" + names[i] + "/cloud_xyz.txt");
			//for (int j = 0; j < share_data->cloud_ground_truth->points.size(); j++) {
			//	fout_xyz << share_data->cloud_ground_truth->points[j].x << " " << share_data->cloud_ground_truth->points[j].y << " " << share_data->cloud_ground_truth->points[j].z << "\n";
			//}
			////复制pose文件
			//ofstream fout_json("D:/Data/NeRF_coverage/pvb_dataset/" + names[i] + "/pose.json", std::ios::binary);
			//ifstream fin_json("E:/HRL/NeRF_coverage/Coverage_images/ShapeNet_" + to_string(id_of_batch) + "/" + names[i] + "/5.json", std::ios::binary);
			//fout_json << fin_json.rdbuf();
			//fout_json.close();
			//fin_json.close();
			//复制图片
			for (int j = 0; j < 64; j++) {
				ofstream fout_image("D:/Data/NeRF_coverage/pvb_dataset/" + names[i] + "/rgbaClip_" + to_string(j) + ".png", std::ios::binary);
				ifstream fin_image("E:/HRL/NeRF_coverage/Coverage_images/ShapeNet_" + to_string(id_of_batch) + "/" + names[i] + "/64/rgbaClip_" + to_string(j) + ".png", std::ios::binary);
				fout_image << fin_image.rdbuf();
				fout_image.close();
				fin_image.close();
			}
			ofstream fout_label("D:/Data/NeRF_coverage/pvb_dataset/" + names[i] + "/view_budget.txt");
			fout_label << labels[i];
			fout_label.close();
			ofstream fout_name("D:/Data/NeRF_coverage/pvb_dataset/names_all.txt", ios::app);
			fout_name << names[i] << "\n";
			fout_name.close();
			share_data.reset();
			labeler.reset();
		}
		//按类输出个数
		ofstream fout_label_num("D:/Data/NeRF_coverage/label_class_num.txt");
		for (auto it = name_num_map.begin(); it != name_num_map.end(); it++) {
			fout_label_num << it->first << "\t";
			for (int i = min_coverage_views; i <= max_coverage_views; i++) {
				fout_label_num << it->second[i] << "\t";
			}
			fout_label_num << "\n";
		}
		//输出label排序的名字
		ofstream fout_label_sorted_name("D:/Data/NeRF_coverage/sorted_object_names.txt");
		fout_label_sorted_name << "count_dataset" << "\t" << label_name_set.size() << "\n";
		fout_label_sorted_name << "mean_label" << "\t" << mean_label / label_name_set.size() << "\n";
		fout_label_sorted_name << "min_label" << "\t" << min_label << "\n";
		fout_label_sorted_name << "max_label" << "\t" << max_label << "\n";
		fout_label_sorted_name << "Label" << "\t" << "Object" << "\n";
		for (auto it = label_name_set.begin(); it != label_name_set.end(); it++) {
			fout_label_sorted_name << it->first << "\t" << it->second << "\n";
		}
		fout_label_sorted_name.close();

		//use map to split the dataset
		map<string, multimap<int, string>> label_name_maps;
		for (auto it_name = name_num_map.begin(); it_name != name_num_map.end(); it_name++) {
			string name_str3 = it_name->first;
			for (auto it = label_name_set.begin(); it != label_name_set.end(); it++) {
				if (it->second.substr(0, 3) == name_str3) {
					label_name_maps[name_str3].insert(make_pair(it->first, it->second));
				}
			}
		}

		ofstream fout_train("D:/Data/NeRF_coverage/pvb_dataset/train_split.txt");
		ofstream fout_val("D:/Data/NeRF_coverage/pvb_dataset/val_split.txt");

		int count_val = 0;
		int dataset_size = label_name_set.size();
		vector<int> train_distribution, val_distribution;
		train_distribution.resize(100);
		val_distribution.resize(100);

		//每个物体的类别进行20/80分割
		for (auto it = label_name_maps.begin(); it != label_name_maps.end(); it++) {
			multimap<int, string> label_name_map = it->second;
			//保证每个label至少有一个进训练集
			for (int label = min_coverage_views; label <= max_coverage_views; label++) {
				auto it2 = label_name_map.find(label);
				if (it2 != label_name_map.end()) {
					fout_train << it2->second << "\n";
					train_distribution[label]++;
					label_name_map.erase(it2);
				}
			}
			//每一个类都要80/20分
			for (int label = min_coverage_views; label <= max_coverage_views; label++) {
				vector<string> name_vec;
				auto it2 = label_name_map.find(label);
				if (it2 != label_name_map.end()) {
					int view_num = it2->first;
					while (it2 != label_name_map.end() && it2->first == view_num) {
						name_vec.push_back(it2->second);
						it2++;
					}
					random_shuffle(name_vec.begin(), name_vec.end());
					int count = 1; //已经有一个进训练集了
					for (auto it3 = name_vec.begin(); it3 != name_vec.end(); it3++) {
						if (count < (name_vec.size() + 1) * 0.8) { //已经有一个进训练集了
							fout_train << *it3 << "\n";
							train_distribution[label]++; 
						}
						else {
							fout_val << *it3 << "\n";
							val_distribution[label]++;
						}
						count++;
					}
				}
			}
		}
		fout_train.close();
		fout_val.close();
		//输出训练集和测试集的分布
		ofstream fout_train_distribution("D:/Data/NeRF_coverage/train_distribution.txt");
		ofstream fout_val_distribution("D:/Data/NeRF_coverage/val_distribution.txt");
		for (int i = min_coverage_views; i <= max_coverage_views; i++) {
			fout_train_distribution << i << "\t" << train_distribution[i] << "\n";
			fout_val_distribution << i << "\t" << val_distribution[i] << "\n";
		}
		fout_train_distribution.close();
		fout_val_distribution.close();
	}
	else if (mode == TestObjects) {
		//从文件读取全部标签
		map<string, int> name_label_map;
		ifstream fin_labels("D:/Data/NeRF_coverage/sorted_object_names.txt");
		string line;
		int count = 0;
		while (getline(fin_labels, line)) {
			if (count < 5) {
				count++;
				continue;
			}
			stringstream ss(line);
			int label;
			string name;
			ss >> label >> name;
			name_label_map[name] = label;
		}

		ifstream check_test_objects("D:/Data/NeRF_coverage/test_objects.txt");
		vector<string> test_names_250;
		if (!check_test_objects.is_open()) {
			//读取测试集
			ifstream fin_test("D:/Data/NeRF_coverage/val_split.txt");
			vector<string> test_names;
			while (getline(fin_test, line)) {
				test_names.push_back(line);
			}
			vector<int> val_distribution;
			val_distribution.resize(100);
			for (int i = 0; i < test_names.size(); i++) {
				val_distribution[name_label_map[test_names[i]]]++;
			}
			int val_num = test_names.size();
			//读已经生成的测试物体
			vector<string> test_names_base;
			vector<int> base_distribution;
			base_distribution.resize(100);
			ifstream fin_test_base("D:/Data/NeRF_coverage/test_objects_base.txt");
			if (fin_test_base.is_open()) {
				while (getline(fin_test_base, line)) {
					test_names_base.push_back(line);
				}
				for (int i = 0; i < test_names_base.size(); i++) {
					base_distribution[name_label_map[test_names_base[i]]]++;
				}
			}
			vector<string> test_names_250(test_names_base);
			int count_num = test_names_250.size();
			for (int i = 13; i <= 58; i++) {
				//按val集合分布采样选取测试集中的250个物体
				int num_needed = (int) round(250.0 * val_distribution[i] / val_num);
				num_needed -= base_distribution[i];
				if (num_needed <= 0) {
					continue;
				}
				count_num += num_needed;
				vector<string> name_vec;
				for (int j = 0; j < test_names.size(); j++) {
					if (name_label_map[test_names[j]] == i) {
						name_vec.push_back(test_names[j]);
					}
				}
				random_shuffle(name_vec.begin(), name_vec.end());
				num_needed = min(num_needed, (int)name_vec.size());
				for (int j = 0; j < num_needed; j++) {
					test_names_250.push_back(name_vec[j]);
				}
			}
			if (count_num > 250) {
				cout << "Error: count_num > 250" << endl;
			}
			cout << "count_num: " << count_num << endl;
			//随机选取剩下的
			while (count_num < 250) {
				int index = rand() % test_names.size();
				if (find(test_names_250.begin(), test_names_250.end(), test_names[index]) != test_names_250.end()) {
					continue;
				}
				test_names_250.push_back(test_names[index]);
				count_num++;
			}
			//输出测试集中的250个物体
			ofstream fout_test_objects("D:/Data/NeRF_coverage/test_objects.txt");
			for (auto it = test_names_250.begin(); it != test_names_250.end(); it++) {
				fout_test_objects << *it << "\n";
			}
			fout_test_objects.close();
			//输出测试集中的250个物体的标签分布
			ofstream fout_test_objects_distribution("D:/Data/NeRF_coverage/test_objects_distribution.txt");
			vector<int> test_distribution;
			test_distribution.resize(100);
			for (auto it = test_names_250.begin(); it != test_names_250.end(); it++) {
				test_distribution[name_label_map[*it]]++;
			}
			for (int i = 13; i <= 58; i++) {
				fout_test_objects_distribution << i << "\t" << test_distribution[i] << "\n";
			}
			fout_test_objects_distribution.close();
		}
		else {
			//从文件读取测试集中的250个物体
			string line;
			while (getline(check_test_objects, line)) {
				test_names_250.push_back(line);
			}
		}

		//读取或生成250个物体的psnr和ssim
		vector<int> gt_views;
		vector<double> gt_psnr, gt_ssim;

		vector<int> pvb_views;
		vector<double> pvb_psnr, pvb_ssim;
		
		vector<vector<double>> statistics_psnr, statistics_ssim;
		//众数32，中位数34，均值35
		vector<int> test_Statistic;
		test_Statistic.push_back(32);
		test_Statistic.push_back(34);
		test_Statistic.push_back(35);
		statistics_psnr.resize(test_Statistic.size());
		statistics_ssim.resize(test_Statistic.size());

		string metric_name;
		bool is_all_generated = false;

		for (auto it = test_names_250.begin(); it != test_names_250.end(); it++) {
			string name = *it;
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", name, -1, -1);
			shared_ptr<NBV_Net_Labeler> labeler = NULL;
			if(!is_all_generated) labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);

			//用于测试的视点空间
			ifstream fin_json_gt(share_data->gt_path + "/" + to_string(100) + ".json");
			if (!fin_json_gt.is_open()) {
				share_data->num_of_views = 100;
				//read viewspace again
				ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
				share_data->pt_sphere.clear();
				share_data->pt_sphere.resize(share_data->num_of_views);
				for (int i = 0; i < share_data->num_of_views; i++) {
					share_data->pt_sphere[i].resize(3);
					for (int j = 0; j < 3; j++) {
						fin_sphere >> share_data->pt_sphere[i][j];
					}
				}
				cout << "view space size is: " << share_data->pt_sphere.size() << endl;
				Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
				share_data->pt_norm = pt0.norm();
				//reget viewspace
				labeler->view_space.reset();
				labeler->view_space = make_shared<View_Space>(share_data);
				//get images
				labeler->get_coverage();
			}

			//计算GT的psnr和ssim
			double psnr_gt, ssim_gt;
			int gt_view = name_label_map[name];
			//if (gt_view % 2 == 0) continue;
			gt_views.push_back(gt_view);
			cout << "gt budget is: " << gt_view << endl;
			ifstream check_psnr_ssim_gt(share_data->gt_path + "/" + to_string(gt_view) + ".txt");
			if (!check_psnr_ssim_gt.is_open()) {
				int num_of_coverage_views = gt_view;
				ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
				if (!fin_json.is_open()) {
					share_data->num_of_views = num_of_coverage_views;
					//read viewspace again
					ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
					share_data->pt_sphere.clear();
					share_data->pt_sphere.resize(share_data->num_of_views);
					for (int i = 0; i < share_data->num_of_views; i++) {
						share_data->pt_sphere[i].resize(3);
						for (int j = 0; j < 3; j++) {
							fin_sphere >> share_data->pt_sphere[i][j];
						}
					}
					cout << "view space size is: " << share_data->pt_sphere.size() << endl;
					Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
					share_data->pt_norm = pt0.norm();
					//reget viewspace
					labeler->view_space.reset();
					labeler->view_space = make_shared<View_Space>(share_data);
					//get images
					labeler->get_coverage();
				}
				//渲染GT的psnr和ssim
				labeler->train_by_instantNGP(to_string(num_of_coverage_views));
			}
			//读取GT的psnr和ssim
			ifstream read_psnr_ssim_gt(share_data->gt_path + "/" + to_string(gt_view) + ".txt");
			//从文件读取psnr和ssim
			read_psnr_ssim_gt >> metric_name;
			read_psnr_ssim_gt >> psnr_gt;
			read_psnr_ssim_gt >> metric_name;
			read_psnr_ssim_gt >> ssim_gt;
			//输出结果
			cout << "GT psnr: " << psnr_gt << endl;
			cout << "GT ssim: " << ssim_gt << endl;
			//保存结果
			gt_psnr.push_back(psnr_gt);
			gt_ssim.push_back(ssim_gt);

			//计算众数32，中位数34，均值35的psnr和ssim
			for (int i = 0; i < test_Statistic.size(); i++) {
				double psnr_Statistic, ssim_Statistic;
				ifstream check_psnr_ssim_Statistic(share_data->gt_path + "/"+ to_string(test_Statistic[i]) + ".txt");
				if (!check_psnr_ssim_Statistic.is_open()) {
					//检查视点空间图片是否存在
					int num_of_coverage_views = test_Statistic[i];
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						labeler->view_space.reset();
						labeler->view_space = make_shared<View_Space>(share_data);
						//get images
						labeler->get_coverage();
					}
					//渲染Statistic34的psnr和ssim
					labeler->train_by_instantNGP(to_string(num_of_coverage_views));
				}
				ifstream read_psnr_ssim_Statistic(share_data->gt_path + "/" + to_string(test_Statistic[i]) + ".txt");
				//从文件读取psnr和ssim
				read_psnr_ssim_Statistic >> metric_name;
				read_psnr_ssim_Statistic >> psnr_Statistic;
				read_psnr_ssim_Statistic >> metric_name;
				read_psnr_ssim_Statistic >> ssim_Statistic;
				//输出结果
				cout << "psnr of Statistic" << test_Statistic[i] << " is: " << psnr_Statistic << endl;
				cout << "ssim of Statistic" << test_Statistic[i] << " is: " << ssim_Statistic << endl;
				//保存结果
				statistics_psnr[i].push_back(psnr_Statistic);
				statistics_ssim[i].push_back(ssim_Statistic);
			}

			//检查是否已经生成了视点预算
			ifstream check_view_budget(share_data->pvb_path + "data/log/" + name + ".txt");
			if (!check_view_budget.is_open()) {
				//通过网络获取视点预算
				ifstream fin_json_init_views(share_data->gt_path + "/" + to_string(5) + ".json");
				if (!fin_json_init_views.is_open()) {//5 init view
					share_data->num_of_views = 5;
					//read viewspace again
					ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
					share_data->pt_sphere.clear();
					share_data->pt_sphere.resize(share_data->num_of_views);
					for (int i = 0; i < share_data->num_of_views; i++) {
						share_data->pt_sphere[i].resize(3);
						for (int j = 0; j < 3; j++) {
							fin_sphere >> share_data->pt_sphere[i][j];
						}
					}
					cout << "view space size is: " << share_data->pt_sphere.size() << endl;
					Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
					share_data->pt_norm = pt0.norm();
					//reget viewspace
					labeler->view_space.reset();
					labeler->view_space = make_shared<View_Space>(share_data);
					//get images
					labeler->get_coverage();
				}
				vector<int> init_view_ids;
				init_view_ids.push_back(0);
				init_view_ids.push_back(1);
				init_view_ids.push_back(3);
				share_data->access_directory(share_data->pvb_path + "data/images");
				for (int i = 0; i < init_view_ids.size(); i++) {
					ofstream fout_image(share_data->pvb_path + "data/images/" + to_string(init_view_ids[i]) + ".png", std::ios::binary);
					ifstream fin_image(share_data->gt_path + "/" + to_string(5) + "/rgbaClip_" + to_string(init_view_ids[i]) + ".png", std::ios::binary);
					fout_image << fin_image.rdbuf();
					fout_image.close();
					fin_image.close();
				}
				ofstream fout_ready(share_data->pvb_path + "data/ready_c++.txt");
				fout_ready.close();
				//等待python程序结束
				ifstream fin_over(share_data->pvb_path + "data/ready_py.txt");
				while (!fin_over.is_open()) {
					boost::this_thread::sleep(boost::posix_time::milliseconds(100));
					fin_over.open(share_data->pvb_path + "data/ready_py.txt");
				}
				fin_over.close();
				boost::this_thread::sleep(boost::posix_time::milliseconds(100));
				remove((share_data->pvb_path + "data/ready_py.txt").c_str());
				//读取view budget
				int view_budget = -1;
				ifstream fin_view_budget(share_data->pvb_path + "data/view_budget.txt");
				if (!fin_view_budget.is_open()) {
					cout << "view budget file not found!" << endl;
				}
				fin_view_budget >> view_budget;
				fin_view_budget.close();
				cout << "view budget is: " << view_budget << endl;
				//view_budget写入硬盘
				share_data->access_directory(share_data->pvb_path + "data/log");
				ofstream fout_view_budget(share_data->pvb_path + "data/log/" + name + ".txt");
				fout_view_budget << view_budget;
				fout_view_budget.close();
			}
			//读取view budget
			int view_budget = -1;
			ifstream fin_view_budget(share_data->pvb_path + "data/log/" + name + ".txt");
			fin_view_budget >> view_budget;
			fin_view_budget.close();
			cout << "view budget is: " << view_budget << endl;

			//读取psnr和ssim
			double psnr_PVB, ssim_PVB;
			ifstream check_psnr_ssim_PVB(share_data->gt_path + "/" + to_string(view_budget) + ".txt");
			if (!check_psnr_ssim_PVB.is_open()) {
				//检查视点空间图片是否存在
				int num_of_coverage_views = view_budget;
				ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
				if (!fin_json.is_open()) {
					share_data->num_of_views = num_of_coverage_views;
					//read viewspace again
					ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
					share_data->pt_sphere.clear();
					share_data->pt_sphere.resize(share_data->num_of_views);
					for (int i = 0; i < share_data->num_of_views; i++) {
						share_data->pt_sphere[i].resize(3);
						for (int j = 0; j < 3; j++) {
							fin_sphere >> share_data->pt_sphere[i][j];
						}
					}
					cout << "view space size is: " << share_data->pt_sphere.size() << endl;
					Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
					share_data->pt_norm = pt0.norm();
					//reget viewspace
					labeler->view_space.reset();
					labeler->view_space = make_shared<View_Space>(share_data);
					//get images
					labeler->get_coverage();
				}
				//渲染PVB的psnr和ssim
				labeler->train_by_instantNGP(to_string(num_of_coverage_views));
			}
			ifstream read_psnr_ssim_PVB(share_data->gt_path + "/" + to_string(view_budget) + ".txt");
			//从文件读取psnr和ssim
			read_psnr_ssim_PVB >> metric_name;
			read_psnr_ssim_PVB >> psnr_PVB;
			read_psnr_ssim_PVB >> metric_name;
			read_psnr_ssim_PVB >> ssim_PVB;
			//输出结果
			cout << "psnr of PVB is: " << psnr_PVB << endl;
			cout << "ssim of PVB is: " << ssim_PVB << endl;
			//保存结果
			pvb_views.push_back(view_budget);
			pvb_psnr.push_back(psnr_PVB);
			pvb_ssim.push_back(ssim_PVB);

			labeler.reset();
			share_data.reset();
		}

		//生成路径查询表
		map<int, double> viewNum_pathLen_map;

		shared_ptr<Share_Data> share_data_path = make_shared<Share_Data>("../DefaultConfiguration.yaml", "", -1);
		for (int num_of_coverage_views = 3; num_of_coverage_views <= 100; num_of_coverage_views++) {
			vector<View> views;
			vector<int> view_set_label;
			int now_view_id = -1;
			//read viewspace again
			share_data_path->num_of_views = num_of_coverage_views;
			ifstream fin_sphere(share_data_path->viewspace_path + to_string(share_data_path->num_of_views) + ".txt");
			share_data_path->pt_sphere.clear();
			share_data_path->pt_sphere.resize(share_data_path->num_of_views);
			for (int i = 0; i < share_data_path->num_of_views; i++) {
				share_data_path->pt_sphere[i].resize(3);
				for (int j = 0; j < 3; j++) {
					fin_sphere >> share_data_path->pt_sphere[i][j];
				}
				Eigen::Vector3d pt(share_data_path->pt_sphere[i][0], share_data_path->pt_sphere[i][1], share_data_path->pt_sphere[i][2]);
				views.push_back(View(pt));
				//如果该视角坐标是0 0 1则记录为当前视点, 使用1e-6
				if (fabs(pt[0]) < 1e-6 && fabs(pt[1]) < 1e-6 && fabs(pt[2] - 1) < 1e-6) {
					now_view_id = i;
				}
				view_set_label.push_back(i);
			}
			fin_sphere.close();
			if (now_view_id == -1) {
				cout << "can not find now view id" << endl;
				//now_view_id = share_data->num_of_views - 1;
			}
			cout << "view space size is: " << share_data_path->pt_sphere.size() << endl;

			vector<int> path;
			Global_Path_Planner* global_path_planner = new Global_Path_Planner(share_data_path, views, view_set_label, now_view_id);
			double total_dis = global_path_planner->solve();
			path = global_path_planner->get_path_id_set();

			viewNum_pathLen_map.insert(make_pair(num_of_coverage_views, total_dis * share_data_path->view_space_radius));
			//cout << "total dis is: " << total_dis << endl;
			delete global_path_planner;
		}
		share_data_path.reset();

		//输出到文件
		ofstream pvb_statistic_compare("D:/Data/NeRF_coverage/pvb_statistic_compare.txt");
		pvb_statistic_compare << setprecision(5);
		//GT计算平均值
		pvb_statistic_compare << "gt_view_budget" << "\t" << "movement_cost" << "\t" << "psnr" << "\t" << "ssim" << "\n";
		double mean_gt_view = 0;
		double mean_gt_movement = 0;
		double mean_gt_psnr = 0;
		double mean_gt_ssim = 0;
		double std_gt_view = 0;
		double std_gt_movement = 0;
		double std_gt_psnr = 0;
		double std_gt_ssim = 0;
		for (int i = 0; i < gt_views.size(); i++) {
			mean_gt_view += gt_views[i];
			mean_gt_movement += viewNum_pathLen_map[gt_views[i]];
			mean_gt_psnr += gt_psnr[i];
			mean_gt_ssim += gt_ssim[i];
		}
		mean_gt_view /= gt_views.size();
		mean_gt_movement /= gt_views.size();
		mean_gt_psnr /= gt_views.size();
		mean_gt_ssim /= gt_views.size();
		for (int i = 0; i < gt_views.size(); i++) {
			std_gt_view += (gt_views[i] - mean_gt_view) * (gt_views[i] - mean_gt_view);
			std_gt_movement += (viewNum_pathLen_map[gt_views[i]] - mean_gt_movement) * (viewNum_pathLen_map[gt_views[i]] - mean_gt_movement);
			std_gt_psnr += (gt_psnr[i] - mean_gt_psnr) * (gt_psnr[i] - mean_gt_psnr);
			std_gt_ssim += (gt_ssim[i] - mean_gt_ssim) * (gt_ssim[i] - mean_gt_ssim);
		}
		std_gt_view /= gt_views.size();
		std_gt_movement /= gt_views.size();
		std_gt_psnr /= gt_views.size();
		std_gt_ssim /= gt_views.size();
		std_gt_view = sqrt(std_gt_view);
		std_gt_movement = sqrt(std_gt_movement);
		std_gt_psnr = sqrt(std_gt_psnr);
		std_gt_ssim = sqrt(std_gt_ssim);
		pvb_statistic_compare << mean_gt_view << "±" << std_gt_view << "\t" << mean_gt_movement << "±" << std_gt_movement << "\t" << mean_gt_psnr << "±" << std_gt_psnr << "\t" << mean_gt_ssim << "±" << std_gt_ssim << "\n";
		pvb_statistic_compare << "\n";

		//PVB计算平均值
		pvb_statistic_compare << "pvb_view_budget" << "\t" << "movement_cost" << "\t" << "psnr" << "\t" << "ssim" << "\t" << "diff_movement_cost" << "\t" << "diff_rate_psnr" << "\t" << "diff_rate_ssim" << "\n";
		double mean_pvb_view = 0;
		double mean_pvb_movement = 0;
		double mean_pvb_psnr = 0;
		double mean_pvb_ssim = 0;
		double mean_pvb_diff_movement = 0;
		double mean_pvb_diff_rate_psnr = 0;
		double mean_pvb_diff_rate_ssim = 0;
		double std_pvb_view = 0;
		double std_pvb_movement = 0;
		double std_pvb_psnr = 0;
		double std_pvb_ssim = 0;
		double std_pvb_diff_movement = 0;
		double std_pvb_diff_rate_psnr = 0;
		double std_pvb_diff_rate_ssim = 0;
		for (int i = 0; i < pvb_views.size(); i++) {
			mean_pvb_view += pvb_views[i];
			mean_pvb_movement += viewNum_pathLen_map[pvb_views[i]];
			mean_pvb_psnr += pvb_psnr[i];
			mean_pvb_ssim += pvb_ssim[i];
			mean_pvb_diff_movement += fabs(viewNum_pathLen_map[pvb_views[i]] - viewNum_pathLen_map[gt_views[i]]);
			mean_pvb_diff_rate_psnr += fabs(pvb_psnr[i] - gt_psnr[i]);
			mean_pvb_diff_rate_ssim += fabs(pvb_ssim[i] - gt_ssim[i]);
		}
		mean_pvb_view /= pvb_views.size();
		mean_pvb_movement /= pvb_views.size();
		mean_pvb_psnr /= pvb_psnr.size();
		mean_pvb_ssim /= pvb_ssim.size();
		mean_pvb_diff_movement /= pvb_views.size();
		mean_pvb_diff_rate_psnr /= pvb_psnr.size();
		mean_pvb_diff_rate_ssim /= pvb_ssim.size();
		for (int i = 0; i < pvb_views.size(); i++) {
			std_pvb_view += (pvb_views[i] - mean_pvb_view) * (pvb_views[i] - mean_pvb_view);
			std_pvb_movement += (viewNum_pathLen_map[pvb_views[i]] - mean_pvb_movement) * (viewNum_pathLen_map[pvb_views[i]] - mean_pvb_movement);
			std_pvb_psnr += (pvb_psnr[i] - mean_pvb_psnr) * (pvb_psnr[i] - mean_pvb_psnr);
			std_pvb_ssim += (pvb_ssim[i] - mean_pvb_ssim) * (pvb_ssim[i] - mean_pvb_ssim);
			std_pvb_diff_movement += (fabs(viewNum_pathLen_map[pvb_views[i]] - viewNum_pathLen_map[gt_views[i]]) - mean_pvb_diff_movement) * (fabs(viewNum_pathLen_map[pvb_views[i]] - viewNum_pathLen_map[gt_views[i]]) - mean_pvb_diff_movement);
			std_pvb_diff_rate_psnr += (fabs(pvb_psnr[i] - gt_psnr[i]) - mean_pvb_diff_rate_psnr) * (fabs(pvb_psnr[i] - gt_psnr[i]) - mean_pvb_diff_rate_psnr);
			std_pvb_diff_rate_ssim += (fabs(pvb_ssim[i] - gt_ssim[i]) - mean_pvb_diff_rate_ssim) * (fabs(pvb_ssim[i] - gt_ssim[i]) - mean_pvb_diff_rate_ssim);
		}
		//这里是样本标准差
		std_pvb_view /= (pvb_views.size() - 1);
		std_pvb_movement /= (pvb_views.size() - 1);
		std_pvb_psnr /= (pvb_psnr.size() - 1);
		std_pvb_ssim /= (pvb_ssim.size() - 1);
		std_pvb_diff_movement /= (pvb_views.size() - 1);
		std_pvb_diff_rate_psnr /= (pvb_psnr.size() - 1);
		std_pvb_diff_rate_ssim /= (pvb_ssim.size() - 1);
		std_pvb_view = sqrt(std_pvb_view);
		std_pvb_movement = sqrt(std_pvb_movement);
		std_pvb_psnr = sqrt(std_pvb_psnr);
		std_pvb_ssim = sqrt(std_pvb_ssim);
		std_pvb_diff_movement = sqrt(std_pvb_diff_movement);
		std_pvb_diff_rate_psnr = sqrt(std_pvb_diff_rate_psnr);
		std_pvb_diff_rate_ssim = sqrt(std_pvb_diff_rate_ssim);
		pvb_statistic_compare << mean_pvb_view << "±" << std_pvb_view << "\t" << mean_pvb_movement << "±" << std_pvb_movement << "\t" << mean_pvb_psnr << "±" << std_pvb_psnr << "\t" << mean_pvb_ssim << "±" << std_pvb_ssim << "\t" << mean_pvb_diff_movement << "±" << std_pvb_diff_movement << "\t"	<< mean_pvb_diff_rate_psnr << "±" << std_pvb_diff_rate_psnr << "\t" << mean_pvb_diff_rate_ssim << "±" << std_pvb_diff_rate_ssim << "\n";
		pvb_statistic_compare << "\n";

		//计算平均值和增加/减少比率
		pvb_statistic_compare << "statistics_view_budget" << "\t" << "movement_cost" << "\t" << "psnr" << "\t" << "ssim" << "\t" << "diff_movement_cost" << "\t" << "diff_rate_psnr" << "\t" << "diff_rate_ssim" << "\n";
		for (int i = 0; i < test_Statistic.size(); i++) {
			double mean_movement = 0;
			double mean_psnr = 0;
			double mean_ssim = 0;
			double std_movement = 0;
			double std_psnr = 0;
			double std_ssim = 0;
			double mean_diff_movement = 0;
			double mean_diff_rate_psnr = 0;
			double mean_diff_rate_ssim = 0;
			double std_diff_movement = 0;
			double std_diff_rate_psnr = 0;
			double std_diff_rate_ssim = 0;
			for (int j = 0; j < statistics_psnr[i].size(); j++) {
				mean_movement += viewNum_pathLen_map[test_Statistic[i]];
				mean_psnr += statistics_psnr[i][j];
				mean_ssim += statistics_ssim[i][j];
				mean_diff_movement += fabs(viewNum_pathLen_map[test_Statistic[i]] - viewNum_pathLen_map[gt_views[j]]);
				mean_diff_rate_psnr += fabs(statistics_psnr[i][j] - gt_psnr[j]);
				mean_diff_rate_ssim += fabs(statistics_ssim[i][j] - gt_ssim[j]);
			}
			mean_movement /= statistics_psnr[i].size();
			mean_psnr /= statistics_psnr[i].size();
			mean_ssim /= statistics_ssim[i].size();
			mean_diff_movement /= statistics_psnr[i].size();
			mean_diff_rate_psnr /= statistics_psnr[i].size();
			mean_diff_rate_ssim /= statistics_ssim[i].size();
			for (int j = 0; j < statistics_psnr[i].size(); j++) {
				std_movement += (viewNum_pathLen_map[test_Statistic[i]] - mean_movement) * (viewNum_pathLen_map[test_Statistic[i]] - mean_movement);
				std_psnr += (statistics_psnr[i][j] - mean_psnr) * (statistics_psnr[i][j] - mean_psnr);
				std_ssim += (statistics_ssim[i][j] - mean_ssim) * (statistics_ssim[i][j] - mean_ssim);
				std_diff_movement += (fabs(viewNum_pathLen_map[test_Statistic[i]] - viewNum_pathLen_map[gt_views[j]]) - mean_diff_movement) * (fabs(viewNum_pathLen_map[test_Statistic[i]] - viewNum_pathLen_map[gt_views[j]]) - mean_diff_movement);
				std_diff_rate_psnr += (fabs(statistics_psnr[i][j] - gt_psnr[j]) - mean_diff_rate_psnr) * (fabs(statistics_psnr[i][j] - gt_psnr[j]) - mean_diff_rate_psnr);
				std_diff_rate_ssim += (fabs(statistics_ssim[i][j] - gt_ssim[j]) - mean_diff_rate_ssim) * (fabs(statistics_ssim[i][j] - gt_ssim[j]) - mean_diff_rate_ssim);
			}
			//这里是样本标准差
			std_movement /= (statistics_psnr[i].size() - 1);
			std_psnr /= (statistics_psnr[i].size() - 1);
			std_ssim /= (statistics_ssim[i].size() - 1);
			std_diff_movement /= (statistics_psnr[i].size() - 1);
			std_diff_rate_psnr /= (statistics_psnr[i].size() - 1);
			std_diff_rate_ssim /= (statistics_ssim[i].size() - 1);
			std_movement = sqrt(std_movement);
			std_psnr = sqrt(std_psnr);
			std_ssim = sqrt(std_ssim);
			std_diff_movement = sqrt(std_diff_movement);
			std_diff_rate_psnr = sqrt(std_diff_rate_psnr);
			std_diff_rate_ssim = sqrt(std_diff_rate_ssim);
			pvb_statistic_compare << test_Statistic[i] << "\t" << mean_movement << "±" << std_movement << "\t" << mean_psnr << "±" << std_psnr << "\t" << mean_ssim << "±" << std_ssim << "\t" << mean_diff_movement << "±" << std_diff_movement << "\t" << mean_diff_rate_psnr << "±" << std_diff_rate_psnr << "\t" << mean_diff_rate_ssim << "±" << std_diff_rate_ssim << "\n";
		}
		pvb_statistic_compare << "\n";

		//保存原始数据
		pvb_statistic_compare << "object" << "\t" << "view_budget(gt,pvb,mode,median,mean)" << "\t" << "movement_cost" << "\t" << "psnr" << "\t" << "ssim" << "\n";
		for (int i = 0; i < test_names_250.size(); i++) {
			pvb_statistic_compare << test_names_250[i] << "\t" << gt_views[i] <<  "\t" << viewNum_pathLen_map[gt_views[i]] << "\t" << gt_psnr[i] << "\t" << gt_ssim[i] << "\n";
			pvb_statistic_compare << test_names_250[i] << "\t" << pvb_views[i] << "\t" << viewNum_pathLen_map[pvb_views[i]] << "\t" << pvb_psnr[i] << "\t" << pvb_ssim[i] << "\n";
			for (int j = 0; j < test_Statistic.size(); j++) {
				pvb_statistic_compare << test_names_250[i] << "\t" << test_Statistic[j] << "\t" << viewNum_pathLen_map[test_Statistic[j]] << "\t" << statistics_psnr[j][i] << "\t" << statistics_ssim[j][i] << "\n";
			}
		}
		pvb_statistic_compare << "\n";

		pvb_statistic_compare.close();
	}
	else if (mode == ShapeNetPreProcess) {
		map<string,string> id2name;
		id2name["04379243"] = "table";
		id2name["02958343"] = "car";
		id2name["03001627"] = "chair";
		id2name["02691156"] = "airplane";
		id2name["04256520"] = "sofa";
		id2name["04090263"] = "rifle";
		id2name["03636649"] = "lamp";
		id2name["04530566"] = "watercraft";
		id2name["02828884"] = "bench";
		id2name["03691459"] = "loudspeaker";
		id2name["02933112"] = "cabinet";
		id2name["03211117"] = "display";
		id2name["04401088"] = "telephone";
		id2name["02924116"] = "bus";
		id2name["02808440"] = "bathtub";
		id2name["03467517"] = "guitar";
		id2name["03325088"] = "faucet";
		id2name["03046257"] = "clock";
		id2name["03991062"] = "flowerpot";
		id2name["03593526"] = "jar";
		shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", "", -1);
		share_data->access_directory("../../3D_models/ShapeNet/");
		ofstream fout_names("../../3D_models/ShapeNet_names.txt");
		for (int i = 0; i < names.size(); i++) {
			string model_path = share_data->shape_net + names[i];
			vector<string> models = getFilesList(model_path);
			cout << id2name[names[i]] + " model number is " << models.size() << endl;
			int textured_num = 0;
			for (int j = 0; j < models.size(); j++) {
				string points_full_path = "E" + models[j] + "/model_normalized_sample.ply";
				ifstream points_file(points_full_path);
				if (points_file.is_open()) {
					ifstream mix_mesh_file_check("../../3D_models/ShapeNet/" + id2name[names[i]] + to_string(textured_num) + ".ply");
					if (!mix_mesh_file_check.is_open()) {
						cout << id2name[names[i]] << " " << models[j] << " is textured, now working. " << endl;
						//读取points文件的顶点数和面数
						string points_line;
						int points_vertex_num = 0;
						int points_face_num = 0;
						getline(points_file, points_line);
						while (points_line != "end_header") {
							if (points_line.find("element vertex") != string::npos) {
								points_vertex_num = stoi(points_line.substr(15));
							}
							else if (points_line.find("element face") != string::npos) {
								points_face_num = stoi(points_line.substr(13));
							}
							getline(points_file, points_line);
						}
						//更改头文件points
						ofstream mix_mesh_file("../../3D_models/ShapeNet/" + id2name[names[i]] + to_string(textured_num) + ".ply");
						mix_mesh_file << "ply\n";
						mix_mesh_file << "format ascii 1.0\n";
						mix_mesh_file << "element vertex " << points_vertex_num << "\n";
						mix_mesh_file << "property float x\n";
						mix_mesh_file << "property float y\n";
						mix_mesh_file << "property float z\n";
						mix_mesh_file << "property uchar red\n";
						mix_mesh_file << "property uchar green\n";
						mix_mesh_file << "property uchar blue\n";
						mix_mesh_file << "end_header\n";
						for (int k = 0; k < points_vertex_num; k++) {
							getline(points_file, points_line);
							//读取顶点坐标和颜色
							vector<double> points_xyz(3);
							vector<int> points_rgba(4);
							stringstream ss(points_line);
							ss >> points_xyz[0] >> points_xyz[1] >> points_xyz[2] >> points_rgba[0] >> points_rgba[1] >> points_rgba[2] >> points_rgba[3];
							//写入新的ply文件
							mix_mesh_file << points_xyz[0] << " " << points_xyz[1] << " " << points_xyz[2] << " ";
							if (points_rgba[0] == 255 && points_rgba[1] == 255 && points_rgba[2] == 255) {
								mix_mesh_file << "250 250 250\n";
							}
							else {
								mix_mesh_file << points_rgba[0] << " " << points_rgba[1] << " " << points_rgba[2] << "\n";
							}
						}
						mix_mesh_file.close();

						//删除读取的mesh和points文件
						//remove(points_full_path.c_str());
					}

					fout_names << id2name[names[i]] + to_string(textured_num) << endl;
					textured_num++;
				}
				else {
					continue;
				}

			}
			cout << id2name[names[i]] + " textured number is " << textured_num << endl;
		}
		fout_names.close();
	}
	else if (mode == GetCleanData) {
		map<string,int> obj_name;
		obj_name.insert(make_pair("table", 0));
		obj_name.insert(make_pair("car", 0));
		obj_name.insert(make_pair("chair", 0));
		obj_name.insert(make_pair("airplane", 0));
		obj_name.insert(make_pair("sofa", 0));
		obj_name.insert(make_pair("rifle", 0));
		obj_name.insert(make_pair("lamp", 0));
		obj_name.insert(make_pair("watercraft", 0));
		obj_name.insert(make_pair("bench", 0));
		obj_name.insert(make_pair("loudspeaker", 0));
		obj_name.insert(make_pair("cabinet", 0));
		obj_name.insert(make_pair("display", 0));
		obj_name.insert(make_pair("telephone", 0));
		obj_name.insert(make_pair("bus", 0));
		obj_name.insert(make_pair("bathtub", 0));
		obj_name.insert(make_pair("guitar", 0));
		obj_name.insert(make_pair("faucet", 0));
		obj_name.insert(make_pair("clock", 0));
		obj_name.insert(make_pair("flowerpot", 0));
		obj_name.insert(make_pair("jar", 0));
		ofstream fout_clean_names("../../3D_models/clean_names.txt");
		int clean_count = 0;
		for (int i = 0; i < names.size(); i++) {
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1);
			ifstream size_reader(share_data->gt_path + "/size.txt");
			if (!size_reader.is_open()) {
				continue;
			}
			double object_size;
			size_reader >> object_size;
			size_reader.close();
			if (object_size > 0.070 && object_size < 0.120) {
				fout_clean_names << names[i] << '\n';
				for (auto it = obj_name.begin(); it != obj_name.end(); it++) {
					if (names[i].find(it->first) != string::npos) {
						it->second++;
						break;
					}
				}
				share_data->access_directory(share_data->pre_path + "Coverage_images/ShapeNet_" + to_string(clean_count / 3000) + "/" + share_data->name_of_pcd + "/");
				ofstream size_writer(share_data->pre_path + "Coverage_images/ShapeNet_"+ to_string(clean_count / 3000) + "/" + share_data->name_of_pcd + "/size.txt");
				size_writer << object_size;
				size_writer.close();
				clean_count++;
				/*
				share_data->access_directory(share_data->pre_path + "Coverage_images_Clean/ShapeNet/" + share_data->name_of_pcd + "/");
				ofstream size_writer_clean(share_data->pre_path + "Coverage_images_Clean/ShapeNet/" + share_data->name_of_pcd + "/size.txt");
				size_writer_clean << object_size;
				size_writer_clean.close();
				*/
			}
		}
		for (auto it = obj_name.begin(); it != obj_name.end(); it++) {
			cout << it->first << " " << it->second << endl;
		}
		fout_clean_names.close();
	}
	else if (mode == GetPathPlan) {
		shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", "", -1);
		//for (int num_of_coverage_views = 100; num_of_coverage_views <= 100; num_of_coverage_views+=16) {
		for (int num_of_coverage_views = 3; num_of_coverage_views <= 100; num_of_coverage_views++) {
			vector<View> views;
			vector<int> view_set_label;
			int now_view_id = -1;
			//read viewspace again
			share_data->num_of_views = num_of_coverage_views;
			ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
			share_data->pt_sphere.clear();
			share_data->pt_sphere.resize(share_data->num_of_views);
			for (int i = 0; i < share_data->num_of_views; i++) {
				share_data->pt_sphere[i].resize(3);
				for (int j = 0; j < 3; j++) {
					fin_sphere >> share_data->pt_sphere[i][j];
				}
				Eigen::Vector3d pt(share_data->pt_sphere[i][0], share_data->pt_sphere[i][1], share_data->pt_sphere[i][2]);
				views.push_back(View(pt));
				//如果该视角坐标是0 0 1则记录为当前视点, 使用1e-6
				if (fabs(pt[0]) < 1e-6 && fabs(pt[1]) < 1e-6 && fabs(pt[2] - 1) < 1e-6) {
					now_view_id = i;
				}
				view_set_label.push_back(i);
			}
			fin_sphere.close();
			if (now_view_id == -1) {
				cout << "can not find now view id" << endl;
				//now_view_id = share_data->num_of_views - 1;
			}
			cout << "view space size is: " << share_data->pt_sphere.size() << endl;

			vector<int> path;
			Global_Path_Planner* global_path_planner = new Global_Path_Planner(share_data, views, view_set_label, now_view_id);
			double total_dis = global_path_planner->solve();
			path = global_path_planner->get_path_id_set();

			cout << "total dis is: " << total_dis << endl;
			delete global_path_planner;
			/*
			//测试每一个终点
			double min_total_dis = 1e10;
			int min_z_view_id = 0;
			for (int end_view_id = 0; end_view_id < share_data->num_of_views; end_view_id++) {
				if (end_view_id != now_view_id) {
					Global_Path_Planner* global_path_planner = new Global_Path_Planner(share_data, views, view_set_label, now_view_id, end_view_id);
					double total_dis = global_path_planner->solve();
					if (total_dis <= min_total_dis + 1e-6) {
						min_total_dis = total_dis;
						if (views[min_z_view_id].init_pos(2) >= views[end_view_id].init_pos(2) - 1e-6) {
							min_z_view_id = end_view_id;
							path = global_path_planner->get_path_id_set();
						}
					}
					delete global_path_planner;
				}
			}
			cout << "min total dis is: " << min_total_dis << endl;
			cout << "min z view id is: " << min_z_view_id << " z value is "<< views[min_z_view_id].init_pos(2) << endl;
			cout << "path size is: " << path.size() << endl;
			*/

			/*
			//生成下一最短距离视点的路径
			path.clear();
			path.push_back(now_view_id);
			int now_view_id_ = now_view_id;
			int now_view_label = 0;
			while (path.size() < share_data->num_of_views) {
				int next_view_id = -1;
				double min_dis = 1e10;
				for (int i = 0; i < share_data->num_of_views; i++) {
					if (i == now_view_id_) {
						continue;
					}
					if (find(path.begin(), path.end(), i) != path.end()) {
						continue;
					}
					double dis = (views[i].init_pos - views[now_view_id_].init_pos).norm();
					if (dis < min_dis) {
						min_dis = dis;
						next_view_id = i;
					}
				}
				path.push_back(next_view_id);
				now_view_id_ = next_view_id;
			}
			double total_dis = 0;
			for (int i = 0; i < path.size() - 1; i++) {
				pair<int, double> local_path = get_local_path(views[path[i]].init_pos.eval(), views[path[i + 1]].init_pos.eval(), Eigen::Vector3d(1e-10, 1e-10, 1e-10), 0.15 / 0.3 * views[0].init_pos.norm());
				total_dis += local_path.second;
			}
			cout << "total distance is: " << total_dis << endl;
			*/

			/*
			//16个近似螺旋下降
			path.clear();
			path.push_back(now_view_id);
			path.push_back(5);
			path.push_back(1);
			path.push_back(4);
			path.push_back(13);
			path.push_back(2);
			path.push_back(6);
			path.push_back(0);
			path.push_back(7);
			path.push_back(8);
			path.push_back(10);
			path.push_back(14);
			path.push_back(11);
			path.push_back(12);
			path.push_back(3);
			path.push_back(9);
			double total_dis = 0;
			for (int i = 0; i < path.size() - 1; i++) {
				total_dis += (views[path[i]].init_pos - views[path[i + 1]].init_pos).norm();
			}
			cout << "total distance is: " << total_dis << endl;
			*/

			/*
			//顺序连接
			path.clear();
			for (int i = 0; i < views.size() -1; i++) {
				path.push_back(i);
				total_dis += (views[path[i]].init_pos - views[path[i + 1]].init_pos).norm();
			}
			path.push_back(views.size() - 1);
			cout << "total distance is: " << total_dis << endl;
			*/
		
			if (share_data->show) { //显示路径
				pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Path"));
				viewer->setBackgroundColor(255, 255, 255);
				//viewer->addCoordinateSystem(0.3);
				viewer->initCameraParameters();
				for (int i = 0; i < views.size(); i++) {
					//获取视点位姿
					views[i].get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), Eigen::Vector3d(1e-10, 1e-10, 1e-10));
					Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * views[i].pose.inverse()).eval();
					//cout << view_pose_world << endl;
					//画视点
					Eigen::Vector4d X(0.1, 0, 0, 1);
					Eigen::Vector4d Y(0, 0.1, 0, 1);
					Eigen::Vector4d Z(0, 0, 0.1, 1);
					Eigen::Vector4d O(0, 0, 0, 1);
					X = view_pose_world * X;
					Y = view_pose_world * Y;
					Z = view_pose_world * Z;
					O = view_pose_world * O;
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i));
				}

				//shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", "flowerpot280", -1, -1, -1);
				//shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", "display108", -1, -1, -1);
				//shared_ptr<NBV_Net_Labeler> labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
				////把cloud_ground_truth中点变为原来的1.0/0.3倍
				//for (int i = 0; i < share_data->cloud_ground_truth->points.size(); i++) {
				//	share_data->cloud_ground_truth->points[i].x *= 1.0 / 0.3;
				//	share_data->cloud_ground_truth->points[i].y *= 1.0 / 0.3;
				//	share_data->cloud_ground_truth->points[i].z *= 1.0 / 0.3;
				//}
				//viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");

				for (int i = 0; i < path.size(); i++) {
					//画路径
					if (i != path.size() - 1) {
						vector<Eigen::Vector3d> points;
						int num_of_path = get_trajectory_xyz(points, views[path[i]].init_pos, views[path[i + 1]].init_pos, Eigen::Vector3d(1e-10, 1e-10, 1e-10), 0.15 / 0.3 * views[0].init_pos.norm(), 0.1 * views[0].init_pos.norm(), 0.0);
						if (num_of_path == -1) {
							cout << "no path. throw" << endl;
							continue;
						}
						if (num_of_path == -2) {
							//cout << "Line. continue" << endl;
							//continue;
						}
						viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(views[path[i]].init_pos(0), views[path[i]].init_pos(1), views[path[i]].init_pos(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 128, 0, 128, "trajectory" + to_string(i) + to_string(i + 1) + to_string(-1));
						viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(i) + to_string(i + 1) + to_string(-1));
						for (int k = 0; k < points.size() - 1; k++) {
							viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 128, 0, 128, "trajectory" + to_string(i) + to_string(i + 1) + to_string(k));
							viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(i) + to_string(i + 1) + to_string(k));
						}
						points.clear();
						points.shrink_to_fit();
					}
				}
				viewer->spinOnce(100);
				while (!viewer->wasStopped())
				{
					viewer->spinOnce(100);
					boost::this_thread::sleep(boost::posix_time::microseconds(100000));
				}
				viewer->close();
				viewer.reset();
			}

			//保存路径
			ofstream fout_path(share_data->viewspace_path + to_string(share_data->num_of_views) + "_path.txt");
			for (int i = 0; i < path.size(); i++) {
				fout_path << path[i] << '\n';
			}
			fout_path.close();
		}
		share_data.reset();
	}
	else if (mode == ViewPlanning) {
		vector<int> method_ids;
		method_ids.push_back(4);
		method_ids.push_back(0);
		method_ids.push_back(1);
		method_ids.push_back(2);
		method_ids.push_back(3);

		// 可交换视角: 0<->2, 3<->4 或者 <0,2,3,4> 四元群
		vector<vector<int>> init_view_ids_cases;
		vector<int> init_view_ids_case_v1;
		init_view_ids_case_v1.push_back(1);
		vector<int> init_view_ids_case_v2;
		init_view_ids_case_v2.push_back(0);
		init_view_ids_case_v2.push_back(1);
		vector<int> init_view_ids_case_v3;
		init_view_ids_case_v3.push_back(0);
		init_view_ids_case_v3.push_back(1);
		init_view_ids_case_v3.push_back(3);
		vector<int> init_view_ids_case_v4;
		init_view_ids_case_v4.push_back(0);
		init_view_ids_case_v4.push_back(1);
		init_view_ids_case_v4.push_back(2);
		init_view_ids_case_v4.push_back(3);
		vector<int> init_view_ids_case_v5;
		init_view_ids_case_v5.push_back(0);
		init_view_ids_case_v5.push_back(1);
		init_view_ids_case_v5.push_back(2);
		init_view_ids_case_v5.push_back(3);
		init_view_ids_case_v5.push_back(4);
		//init_view_ids_cases.push_back(init_view_ids_case_v1);
		//init_view_ids_cases.push_back(init_view_ids_case_v2);
		init_view_ids_cases.push_back(init_view_ids_case_v3);
		//init_view_ids_cases.push_back(init_view_ids_case_v4);
		//init_view_ids_cases.push_back(init_view_ids_case_v5);

		int num_of_random_test = 1;

		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < method_ids.size(); j++) {
				//保证5-64/100个视点有数据
				vector<View> init_views;
				shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, -1, method_ids[j]);
				shared_ptr<NBV_Net_Labeler> labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
				if (!labeler->object_is_ok_size) {
					cout << "object size is too small. continue" << endl;
					continue;
				}
				{//144/540
					int num_of_coverage_views = share_data->num_of_views;
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						labeler->view_space.reset();
						labeler->view_space = make_shared<View_Space>(share_data);
						//get images
						labeler->get_coverage();
					}
				}
				cout << "init view space images." << endl;
				int num_of_views = share_data->num_of_views;
				for (int num_of_coverage_views = 5; num_of_coverage_views <= 60; num_of_coverage_views++) {
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						labeler->view_space.reset();
						labeler->view_space = make_shared<View_Space>(share_data);
						//get images
						labeler->get_coverage();
					}
				}
				{//100
					int num_of_coverage_views = 100;
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						labeler->view_space.reset();
						labeler->view_space = make_shared<View_Space>(share_data);
						//get images
						labeler->get_coverage();
					}
				}
				{//5 init view
					share_data->num_of_views = 5;
					//read viewspace again
					ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
					share_data->pt_sphere.clear();
					share_data->pt_sphere.resize(share_data->num_of_views);
					for (int i = 0; i < share_data->num_of_views; i++) {
						share_data->pt_sphere[i].resize(3);
						for (int j = 0; j < 3; j++) {
							fin_sphere >> share_data->pt_sphere[i][j];
						}
					}
					cout << "view space size is: " << share_data->pt_sphere.size() << endl;
					Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
					share_data->pt_norm = pt0.norm();
					//reget viewspace
					labeler->view_space.reset();
					labeler->view_space = make_shared<View_Space>(share_data);
					init_views = labeler->view_space->views;
					cout << "init view space with size: " << init_views.size() << endl;
				}
				labeler.reset();
				share_data.reset();
				//NBV测试
				for (int init_case = 0; init_case < init_view_ids_cases.size(); init_case++) {
					for(int random_test_id = 0; random_test_id < num_of_random_test; random_test_id++){
						share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, -1, method_ids[j]);
						labeler = make_shared<NBV_Net_Labeler>(share_data, 0, 0);
						labeler->init_views = init_views;
						int now_view_id = -1;
						for (int i = 0; i < share_data->num_of_views; i++) {
							if (fabs(labeler->view_space->views[i].init_pos(0)) < 1e-6 && fabs(labeler->view_space->views[i].init_pos(1)) < 1e-6 && fabs(labeler->view_space->views[i].init_pos(2) - share_data->view_space_radius) < 1e-6) {
								now_view_id = i;
							}
						}
						if (now_view_id == -1) {
							cout << "can not find now view id" << endl;
						}
						cout << "start view planning." << endl;
						labeler->nbv_loop(now_view_id, init_view_ids_cases[init_case], random_test_id);
						labeler.reset();
						share_data.reset();
					}
				}
			}
		}
	}
	cout << "System over." << endl;
	return 0;
}

/*
Armadillo
Asian_Dragon
Dragon
Stanford_Bunny
Happy_Buddha
Thai_Statue
Lucy
LM1
LM2
LM3
LM4
LM5
LM6
LM7
LM8
LM9
LM10
LM11
LM12
obj_000001
obj_000002
obj_000003
obj_000004
obj_000005
obj_000006
obj_000007
obj_000008
obj_000009
obj_000010
obj_000011
obj_000012
obj_000013
obj_000014
obj_000015
obj_000016
obj_000017
obj_000018
obj_000019
obj_000020
obj_000021
obj_000022
obj_000023
obj_000024
obj_000025
obj_000026
obj_000027
obj_000028
obj_000029
obj_000030
obj_000031
obj_000032
*/

/*
04379243
02958343
03001627
02691156
04256520
04090263
03636649
04530566
02828884
03691459
02933112
03211117
04401088
02924116
02808440
03467517
03325088
03046257
03991062
03593526
*/