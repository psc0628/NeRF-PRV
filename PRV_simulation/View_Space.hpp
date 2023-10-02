#pragma once
#include <iostream> 
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <time.h>
#include <mutex>
#include <unordered_set>
#include <bitset>

#include <opencv2/opencv.hpp>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;

inline double get_random_coordinate(double from, double to) {
	//生成比较随机的0-1随机数并映射到区间[from,to]
	double len = to - from;
	long long x = (long long)rand() * ((long long)RAND_MAX + 1) + (long long)rand();
	long long field = (long long)RAND_MAX * (long long)RAND_MAX + 2 * (long long)RAND_MAX;
	return (double)x / (double)field * len + from;
}

class View
{
public:
	Eigen::Vector3d init_pos;	//初始位置
	Eigen::Matrix4d pose;		//view_i到view_i+1旋转矩阵

	View(Eigen::Vector3d _init_pos) {
		init_pos = _init_pos;
		pose = Eigen::Matrix4d::Identity(4, 4);
	}

	View(const View &other) {
		init_pos = other.init_pos;
		pose = other.pose;
	}

	View& operator=(const View& other) {
		init_pos = other.init_pos;
		pose = other.pose;
		return *this;
	}

	~View() {

	}

	//0 simlarest to previous, 1 y-top
	void get_next_camera_pos(Eigen::Matrix4d now_camera_pose_world, Eigen::Vector3d object_center_world, int type_of_pose = 0) {
		switch (type_of_pose) {
			case 0:
			{
				//归一化乘法
				Eigen::Vector4d object_center_now_camera;
				object_center_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(object_center_world(0), object_center_world(1), object_center_world(2), 1);
				Eigen::Vector4d view_now_camera;
				view_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(init_pos(0), init_pos(1), init_pos(2), 1);
				//定义指向物体为Z+，从上一个相机位置发出射线至当前为X+，计算两个相机坐标系之间的变换矩阵，object与view为上一个相机坐标系下的坐标
				Eigen::Vector3d object(object_center_now_camera(0), object_center_now_camera(1), object_center_now_camera(2));
				Eigen::Vector3d view(view_now_camera(0), view_now_camera(1), view_now_camera(2));
				Eigen::Vector3d Z;	 Z = object - view;	 Z = Z.normalized();
				//注意左右手系，不要弄反了
				Eigen::Vector3d X;	 X = Z.cross(view);	 X = X.normalized();
				Eigen::Vector3d Y;	 Y = Z.cross(X);	 Y = Y.normalized();
				Eigen::Matrix4d T(4, 4);
				T(0, 0) = 1; T(0, 1) = 0; T(0, 2) = 0; T(0, 3) = -view(0);
				T(1, 0) = 0; T(1, 1) = 1; T(1, 2) = 0; T(1, 3) = -view(1);
				T(2, 0) = 0; T(2, 1) = 0; T(2, 2) = 1; T(2, 3) = -view(2);
				T(3, 0) = 0; T(3, 1) = 0; T(3, 2) = 0; T(3, 3) = 1;
				Eigen::Matrix4d R(4, 4);
				R(0, 0) = X(0); R(0, 1) = Y(0); R(0, 2) = Z(0); R(0, 3) = 0;
				R(1, 0) = X(1); R(1, 1) = Y(1); R(1, 2) = Z(1); R(1, 3) = 0;
				R(2, 0) = X(2); R(2, 1) = Y(2); R(2, 2) = Z(2); R(2, 3) = 0;
				R(3, 0) = 0;	R(3, 1) = 0;	R(3, 2) = 0;	R(3, 3) = 1;
				//绕Z轴旋转，使得与上一次旋转计算x轴与y轴夹角最小
				Eigen::Matrix3d Rz_min(Eigen::Matrix3d::Identity(3, 3));
				Eigen::Vector4d x(1, 0, 0, 1);
				Eigen::Vector4d y(0, 1, 0, 1);
				Eigen::Vector4d x_ray(1, 0, 0, 1);
				Eigen::Vector4d y_ray(0, 1, 0, 1);
				x_ray = R.inverse() * T * x_ray;
				y_ray = R.inverse() * T * y_ray;
				double min_y = acos(y(1) * y_ray(1));
				double min_x = acos(x(0) * x_ray(0));
				for (double i = 5; i < 360; i += 5) {
					Eigen::Matrix3d rotation;
					rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
						Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
						Eigen::AngleAxisd(i * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
					Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
					Rz(0, 0) = rotation(0, 0); Rz(0, 1) = rotation(0, 1); Rz(0, 2) = rotation(0, 2); Rz(0, 3) = 0;
					Rz(1, 0) = rotation(1, 0); Rz(1, 1) = rotation(1, 1); Rz(1, 2) = rotation(1, 2); Rz(1, 3) = 0;
					Rz(2, 0) = rotation(2, 0); Rz(2, 1) = rotation(2, 1); Rz(2, 2) = rotation(2, 2); Rz(2, 3) = 0;
					Rz(3, 0) = 0;			   Rz(3, 1) = 0;			  Rz(3, 2) = 0;			     Rz(3, 3) = 1;
					Eigen::Vector4d x_ray(1, 0, 0, 1);
					Eigen::Vector4d y_ray(0, 1, 0, 1);
					x_ray = (R * Rz).inverse() * T * x_ray;
					y_ray = (R * Rz).inverse() * T * y_ray;
					double cos_y = acos(y(1) * y_ray(1));
					double cos_x = acos(x(0) * x_ray(0));
					if (cos_y < min_y) {
						Rz_min = rotation.eval();
						min_y = cos_y;
						min_x = cos_x;
					}
					else if (fabs(cos_y - min_y) < 1e-6 && cos_x < min_x) {
						Rz_min = rotation.eval();
						min_y = cos_y;
						min_x = cos_x;
					}
				}
				Eigen::Vector3d eulerAngle = Rz_min.eulerAngles(0, 1, 2);
				//cout << "Rotate getted with angel " << eulerAngle(0)<<","<< eulerAngle(1) << "," << eulerAngle(2)<<" and l2 "<< min_l2 << endl;
				Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
				Rz(0, 0) = Rz_min(0, 0); Rz(0, 1) = Rz_min(0, 1); Rz(0, 2) = Rz_min(0, 2); Rz(0, 3) = 0;
				Rz(1, 0) = Rz_min(1, 0); Rz(1, 1) = Rz_min(1, 1); Rz(1, 2) = Rz_min(1, 2); Rz(1, 3) = 0;
				Rz(2, 0) = Rz_min(2, 0); Rz(2, 1) = Rz_min(2, 1); Rz(2, 2) = Rz_min(2, 2); Rz(2, 3) = 0;
				Rz(3, 0) = 0;			 Rz(3, 1) = 0;			  Rz(3, 2) = 0;			   Rz(3, 3) = 1;
				pose = ((R * Rz).inverse() * T).eval();
				//pose = (R.inverse() * T).eval();
			}
			break;
			case 1:
			{
				//归一化乘法
				Eigen::Vector4d object_center_now_camera;
				object_center_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(object_center_world(0), object_center_world(1), object_center_world(2), 1);
				Eigen::Vector4d view_now_camera;
				view_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(init_pos(0), init_pos(1), init_pos(2), 1);
				//定义指向物体为Z+，从上一个相机位置发出射线至当前为X+，计算两个相机坐标系之间的变换矩阵，object与view为上一个相机坐标系下的坐标
				Eigen::Vector3d object(object_center_now_camera(0), object_center_now_camera(1), object_center_now_camera(2));
				Eigen::Vector3d view(view_now_camera(0), view_now_camera(1), view_now_camera(2));
				Eigen::Vector3d Z;	 Z = object - view;	 Z = Z.normalized();
				//注意左右手系，不要弄反了
				Eigen::Vector3d X;	 X = Z.cross(view);	 X = X.normalized();
				Eigen::Vector3d Y;	 Y = Z.cross(X);	 Y = Y.normalized();
				Eigen::Matrix4d T(4, 4);
				T(0, 0) = 1; T(0, 1) = 0; T(0, 2) = 0; T(0, 3) = -view(0);
				T(1, 0) = 0; T(1, 1) = 1; T(1, 2) = 0; T(1, 3) = -view(1);
				T(2, 0) = 0; T(2, 1) = 0; T(2, 2) = 1; T(2, 3) = -view(2);
				T(3, 0) = 0; T(3, 1) = 0; T(3, 2) = 0; T(3, 3) = 1;
				Eigen::Matrix4d R(4, 4);
				R(0, 0) = X(0); R(0, 1) = Y(0); R(0, 2) = Z(0); R(0, 3) = 0;
				R(1, 0) = X(1); R(1, 1) = Y(1); R(1, 2) = Z(1); R(1, 3) = 0;
				R(2, 0) = X(2); R(2, 1) = Y(2); R(2, 2) = Z(2); R(2, 3) = 0;
				R(3, 0) = 0;	R(3, 1) = 0;	R(3, 2) = 0;	R(3, 3) = 1;
				//绕Z轴旋转，使得y轴最高
				Eigen::Matrix3d Rz_min(Eigen::Matrix3d::Identity(3, 3));
				Eigen::Vector4d y_highest = (now_camera_pose_world * R * T * Eigen::Vector4d(0, 1, 0, 1)).eval();
				for (double i = 5; i < 360; i += 5) {
					Eigen::Matrix3d rotation;
					rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
						Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
						Eigen::AngleAxisd(i * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
					Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
					Rz(0, 0) = rotation(0, 0); Rz(0, 1) = rotation(0, 1); Rz(0, 2) = rotation(0, 2); Rz(0, 3) = 0;
					Rz(1, 0) = rotation(1, 0); Rz(1, 1) = rotation(1, 1); Rz(1, 2) = rotation(1, 2); Rz(1, 3) = 0;
					Rz(2, 0) = rotation(2, 0); Rz(2, 1) = rotation(2, 1); Rz(2, 2) = rotation(2, 2); Rz(2, 3) = 0;
					Rz(3, 0) = 0;			   Rz(3, 1) = 0;			  Rz(3, 2) = 0;			     Rz(3, 3) = 1;
					Eigen::Vector4d y_now = (now_camera_pose_world * R * Rz * T * Eigen::Vector4d(0, 1, 0, 1)).eval();
					if (y_now(2) > y_highest(2)) {
						Rz_min = rotation.eval();
						y_highest = y_now;
					}
				}
				Eigen::Vector3d eulerAngle = Rz_min.eulerAngles(0, 1, 2);
				//cout << "Rotate getted with angel " << eulerAngle(0)<<","<< eulerAngle(1) << "," << eulerAngle(2)<<" and l2 "<< min_l2 << endl;
				Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
				Rz(0, 0) = Rz_min(0, 0); Rz(0, 1) = Rz_min(0, 1); Rz(0, 2) = Rz_min(0, 2); Rz(0, 3) = 0;
				Rz(1, 0) = Rz_min(1, 0); Rz(1, 1) = Rz_min(1, 1); Rz(1, 2) = Rz_min(1, 2); Rz(1, 3) = 0;
				Rz(2, 0) = Rz_min(2, 0); Rz(2, 1) = Rz_min(2, 1); Rz(2, 2) = Rz_min(2, 2); Rz(2, 3) = 0;
				Rz(3, 0) = 0;			 Rz(3, 1) = 0;			  Rz(3, 2) = 0;			   Rz(3, 3) = 1;
				pose = ((R * Rz).inverse() * T).eval();
				//pose = (R.inverse() * T).eval();
			}
			break;
		}

	}

};

#define ErrorPath -2
#define WrongPath -1
#define LinePath 0
#define CirclePath 1
//return path mode and length from M to N under an circle obstacle with radius r
pair<int, double> get_local_path(Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double r) {
	double x0, y0, z0, x1, y1, z1, x2, y2, z2, a, b, c, delta, t3, t4, x3, y3, z3, x4, y4, z4;
	x1 = M(0), y1 = M(1), z1 = M(2);
	x2 = N(0), y2 = N(1), z2 = N(2);
	x0 = O(0), y0 = O(1), z0 = O(2);
	//计算直线MN与球O-r的交点PQ
	a = pow2(x2 - x1) + pow2(y2 - y1) + pow2(z2 - z1);
	b = 2.0 * ((x2 - x1) * (x1 - x0) + (y2 - y1) * (y1 - y0) + (z2 - z1) * (z1 - z0));
	c = pow2(x1 - x0) + pow2(y1 - y0) + pow2(z1 - z0) - pow2(r);
	delta = pow2(b) - 4.0 * a * c;
	//cout << delta << endl;
	if (delta <= 0) {//delta <= 0
		//如果没有交点或者一个交点，就可以画直线过去
		double d = (N - M).norm();
		//cout << "d: " << d << endl;
		return make_pair(LinePath, d);
	}
	else {
		//如果需要穿过球体，则沿着球表面行动
		t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
			//球外两点，直接过去
			double d = (N - M).norm();
			//cout << "d: " << d << endl;
			return make_pair(LinePath, d);
		}
		else if ((t3 < 0 || t3 > 1) || (t4 < 0 || t4 > 1)) {
			//起点或终点在障碍物球内
			return make_pair(WrongPath, 1e10);
		}
		if (t3 > t4) {
			double temp = t3;
			t3 = t4;
			t4 = temp;
		}
		x3 = (x2 - x1) * t3 + x1;
		y3 = (y2 - y1) * t3 + y1;
		z3 = (z2 - z1) * t3 + z1;
		Eigen::Vector3d P(x3, y3, z3);
		//cout << "P: " << x3 << "," << y3 << "," << z3 << endl;
		x4 = (x2 - x1) * t4 + x1;
		y4 = (y2 - y1) * t4 + y1;
		z4 = (z2 - z1) * t4 + z1;
		Eigen::Vector3d Q(x4, y4, z4);
		//cout << "Q: " << x4 << "," << y4 << "," << z4 << endl;
		//MON平面方程
		double A, B, C, D, X1, X2, Y1, Y2, Z1, Z2;
		X1 = x3 - x0; X2 = x4 - x0;
		Y1 = y3 - y0; Y2 = y4 - y0;
		Z1 = z3 - z0; Z2 = z4 - z0;
		A = Y1 * Z2 - Y2 * Z1;
		B = Z1 * X2 - Z2 * X1;
		C = X1 * Y2 - X2 * Y1;
		D = -A * x0 - B * y0 - C * z0;
		//D = -(x0 * Y1 * Z2 + X1 * Y2 * z0 + X2 * y0 * Z1 - X2 * Y1 * z0 - X1 * y0 * Z2 - x0 * Y2 * Z1);
		//计算参数方程中P,Q的参数值
		double theta3, theta4, flag, MP, QN, L, d;
		double sin_theta3, sin_theta4;
		sin_theta3 = -(z3 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta3 = asin(sin_theta3);
		if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
		if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		double x3_theta3, y3_theta3;
		x3_theta3 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		y3_theta3 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		//cout << x3_theta3 << " " << y3_theta3 << " " << theta3 << endl;
		if (fabs(x3 - x3_theta3) > 1e-6 || fabs(y3 - y3_theta3) > 1e-6) {
			theta3 = acos(-1.0) - theta3;
			if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
			if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		}
		sin_theta4 = -(z4 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta4 = asin(sin_theta4);
		if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
		if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		double x4_theta4, y4_theta4;
		x4_theta4 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		y4_theta4 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		//cout << x4_theta4 << " " << y4_theta4 << " " << theta4 << endl;
		if (fabs(x4 - x4_theta4) > 1e-6 || fabs(y4 - y4_theta4) > 1e-6) {
			theta4 = acos(-1.0) - theta4;
			if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
			if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		}
		//cout << "theta3: " << theta3 << endl;
		//cout << "theta4: " << theta4 << endl;
		if (theta3 < theta4) flag = 1;
		else flag = -1;
		MP = (M - P).norm();
		QN = (Q - N).norm();
		L = fabs(theta3 - theta4) * r;
		//cout << "L: " << L << endl;
		d = MP + L + QN;
		//cout << "d: " << d << endl;
		return make_pair(CirclePath, d);
	}
	//未定义行为
	return make_pair(ErrorPath, 1e10);
}

double get_trajectory_xyz(vector<Eigen::Vector3d>& points, Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double predicted_size, double distanse_of_pre_move, double camera_to_object_dis) {
	int num_of_path = -1;
	double x0, y0, z0, x1, y1, z1, x2, y2, z2, a, b, c, delta, t3, t4, x3, y3, z3, x4, y4, z4, r;
	x1 = M(0), y1 = M(1), z1 = M(2);
	x2 = N(0), y2 = N(1), z2 = N(2);
	x0 = O(0), y0 = O(1), z0 = O(2);
	r = predicted_size + camera_to_object_dis;
	//计算直线MN与球O-r的交点PQ
	a = pow2(x2 - x1) + pow2(y2 - y1) + pow2(z2 - z1);
	b = 2.0 * ((x2 - x1) * (x1 - x0) + (y2 - y1) * (y1 - y0) + (z2 - z1) * (z1 - z0));
	c = pow2(x1 - x0) + pow2(y1 - y0) + pow2(z1 - z0) - pow2(r);
	delta = pow2(b) - 4.0 * a * c;
	//cout << delta << endl;
	if (delta <= 0) {//delta <= 0
		//如果没有交点或者一个交点，就可以画直线过去
		double d = (N - M).norm();
		//cout << "d: " << d << endl;
		num_of_path = (int)(d / distanse_of_pre_move) + 1;
		//cout << "num_of_path: " << num_of_path << endl;
		double t_pre_point = d / num_of_path;
		for (int i = 1; i <= num_of_path; i++) {
			double di = t_pre_point * i;
			//cout << "di: " << di << endl;
			double xi, yi, zi;
			xi = (x2 - x1) * (di / d) + x1;
			yi = (y2 - y1) * (di / d) + y1;
			zi = (z2 - z1) * (di / d) + z1;
			points.push_back(Eigen::Vector3d(xi, yi, zi));
		}
		return -2;
	}
	else {
		//如果需要穿过球体，则沿着球表面行动
		t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
			//球外两点，直接过去
			double d = (N - M).norm();
			//cout << "d: " << d << endl;
			num_of_path = (int)(d / distanse_of_pre_move) + 1;
			//cout << "num_of_path: " << num_of_path << endl;
			double t_pre_point = d / num_of_path;
			for (int i = 1; i <= num_of_path; i++) {
				double di = t_pre_point * i;
				//cout << "di: " << di << endl;
				double xi, yi, zi;
				xi = (x2 - x1) * (di / d) + x1;
				yi = (y2 - y1) * (di / d) + y1;
				zi = (z2 - z1) * (di / d) + z1;
				points.push_back(Eigen::Vector3d(xi, yi, zi));
			}
			return num_of_path;
		}
		else if ((t3 < 0 || t3 > 1) || (t4 < 0 || t4 > 1)) {
			cout << "has one viewport in circle. check?" << endl;
			return -1;
		}
		if (t3 > t4) {
			double temp = t3;
			t3 = t4;
			t4 = temp;
		}
		x3 = (x2 - x1) * t3 + x1;
		y3 = (y2 - y1) * t3 + y1;
		z3 = (z2 - z1) * t3 + z1;
		Eigen::Vector3d P(x3, y3, z3);
		//cout << "P: " << x3 << "," << y3 << "," << z3 << endl;
		x4 = (x2 - x1) * t4 + x1;
		y4 = (y2 - y1) * t4 + y1;
		z4 = (z2 - z1) * t4 + z1;
		Eigen::Vector3d Q(x4, y4, z4);
		//cout << "Q: " << x4 << "," << y4 << "," << z4 << endl;
		//MON平面方程
		double A, B, C, D, X1, X2, Y1, Y2, Z1, Z2;
		X1 = x3 - x0; X2 = x4 - x0;
		Y1 = y3 - y0; Y2 = y4 - y0;
		Z1 = z3 - z0; Z2 = z4 - z0;
		A = Y1 * Z2 - Y2 * Z1;
		B = Z1 * X2 - Z2 * X1;
		C = X1 * Y2 - X2 * Y1;
		D = -A * x0 - B * y0 - C * z0;
		//D = -(x0 * Y1 * Z2 + X1 * Y2 * z0 + X2 * y0 * Z1 - X2 * Y1 * z0 - X1 * y0 * Z2 - x0 * Y2 * Z1);
		//计算参数方程中P,Q的参数值
		double theta3, theta4, flag, MP, QN, L, d;
		double sin_theta3, sin_theta4;
		sin_theta3 = -(z3 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta3 = asin(sin_theta3);
		if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
		if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		double x3_theta3, y3_theta3;
		x3_theta3 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		y3_theta3 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		//cout << x3_theta3 << " " << y3_theta3 << " " << theta3 << endl;
		if (fabs(x3 - x3_theta3) > 1e-6 || fabs(y3 - y3_theta3) > 1e-6) {
			theta3 = acos(-1.0) - theta3;
			if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
			if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		}
		sin_theta4 = -(z4 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta4 = asin(sin_theta4);
		if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
		if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		double x4_theta4, y4_theta4;
		x4_theta4 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		y4_theta4 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		//cout << x4_theta4 << " " << y4_theta4 << " " << theta4 << endl;
		if (fabs(x4 - x4_theta4) > 1e-6 || fabs(y4 - y4_theta4) > 1e-6) {
			theta4 = acos(-1.0) - theta4;
			if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
			if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		}
		//cout << "theta3: " << theta3 << endl;
		//cout << "theta4: " << theta4 << endl;
		if (theta3 < theta4) flag = 1;
		else flag = -1;
		MP = (M - P).norm();
		QN = (Q - N).norm();
		L = fabs(theta3 - theta4) * r;
		//cout << "L: " << L << endl;
		d = MP + L + QN;
		//cout << "d: " << d << endl;
		num_of_path = (int)(d / distanse_of_pre_move) + 1;
		//cout << "num_of_path: " << num_of_path << endl;
		double t_pre_point = d / num_of_path;
		bool on_ground = true;
		for (int i = 1; i <= num_of_path; i++) {
			double di = t_pre_point * i;
			//cout << "di: " << di << endl;
			double xi, yi, zi;
			if (di <= MP || di >= MP + L) {
				xi = (x2 - x1) * (di / d) + x1;
				yi = (y2 - y1) * (di / d) + y1;
				zi = (z2 - z1) * (di / d) + z1;
			}
			else {
				double di_theta = di - MP;
				double theta_i = flag * di_theta / r + theta3;
				//cout << "theta_i: " << theta_i << endl;
				xi = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
				yi = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
				zi = z0 - r * sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
				if (zi < 0.05) {
					//如果路径点高度为负值，表示求解的路径是下半部分球体，翻转方向
					on_ground = false;
					break;
				}
			}
			points.push_back(Eigen::Vector3d(xi, yi, zi));
		}
		//return d;
		if (!on_ground) {
			cout << "Another way." << endl;
			L = 2.0 * acos(-1.0) * r - fabs(theta3 - theta4) * r;
			//cout << "L: " << L << endl;
			d = MP + L + QN;
			//cout << "d: " << d << endl;
			num_of_path = (int)(d / distanse_of_pre_move) + 1;
			//cout << "num_of_path: " << num_of_path << endl;
			t_pre_point = d / num_of_path;
			flag = -flag;
			points.clear();
			for (int i = 1; i <= num_of_path; i++) {
				double di = t_pre_point * i;
				//cout << "di: " << di << endl;
				double xi, yi, zi;
				if (di <= MP || di >= MP + L) {
					xi = (x2 - x1) * (di / d) + x1;
					yi = (y2 - y1) * (di / d) + y1;
					zi = (z2 - z1) * (di / d) + z1;
				}
				else {
					double di_theta = di - MP;
					double theta_i = flag * di_theta / r + theta3;
					//cout << "theta_i: " << theta_i << endl;
					xi = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
					yi = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
					zi = z0 - r * sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
				}
				points.push_back(Eigen::Vector3d(xi, yi, zi));
			}
		}
	}
	return num_of_path;
}

class View_Space
{
public:
	int num_of_views;							//视点个数
	vector<View> views;							//空间的采样视点
	Eigen::Vector3d object_center_world;		//物体中心
	double predicted_size;						//物体BBX半径
	Eigen::Matrix4d now_camera_pose_world;		//这次nbv迭代的相机位置
	int occupied_voxels;						
	double map_entropy;	
	double octomap_resolution;
	shared_ptr<octomap::ColorOcTree> octo_model;
	shared_ptr<Share_Data> share_data;

	double check_size(double predicted_size, vector<Eigen::Vector3d>& points) {
		int vaild_points = 0;
		for (auto& ptr : points) {
			if (ptr(0) < object_center_world(0) - predicted_size || ptr(0) > object_center_world(0) + predicted_size) continue;
			if (ptr(1) < object_center_world(1) - predicted_size || ptr(1) > object_center_world(1) + predicted_size) continue;
			if (ptr(2) < object_center_world(2) - predicted_size || ptr(2) > object_center_world(2) + predicted_size) continue;
			vaild_points++;
		}
		return (double)vaild_points / (double)points.size();
	}

	void get_view_space(vector<Eigen::Vector3d>& points) {
		double now_time = clock();
		/*
		//计算物体中心
		Eigen::Vector3d mesh_min_pos = points[0];
		Eigen::Vector3d mesh_max_pos = points[0];
		for (int k = 1; k < points.size(); k++) {
			mesh_min_pos(0) = min(mesh_min_pos(0), points[k](0));
			mesh_min_pos(1) = min(mesh_min_pos(1), points[k](1));
			mesh_min_pos(2) = min(mesh_min_pos(2), points[k](2));
			mesh_max_pos(0) = max(mesh_max_pos(0), points[k](0));
			mesh_max_pos(1) = max(mesh_max_pos(1), points[k](1));
			mesh_max_pos(2) = max(mesh_max_pos(2), points[k](2));
		}
		object_center_world = (mesh_min_pos + mesh_max_pos) / 2;
		*/
		//计算点云质心
		object_center_world = Eigen::Vector3d(0, 0, 0);
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
		//计算最远点
		predicted_size = 0.0;
		for (auto& ptr : points) {
			predicted_size = max(predicted_size, (object_center_world - ptr).norm());
		}
		predicted_size *= 17.0 / 16.0;
		cout << "object's pos is ("<< object_center_world(0) << "," << object_center_world(1) << "," << object_center_world(2) << ") and size is " << predicted_size << endl;
		for (int i = 0; i < share_data->pt_sphere.size(); i++) {
			if (share_data->pt_sphere[i][2] < 0) continue;
			double scale = 1.0 / share_data->pt_norm * share_data->view_space_radius; //predicted_size * sqrt(3) * ?
			View view(Eigen::Vector3d(share_data->pt_sphere[i][0]* scale + object_center_world(0), share_data->pt_sphere[i][1] * scale + object_center_world(1), share_data->pt_sphere[i][2] * scale + object_center_world(2)));
			//cout << share_data->pt_sphere[i][0] * scale << " " << share_data->pt_sphere[i][1] * scale << " " << share_data->pt_sphere[i][2] * scale << endl;
			views.push_back(view);
		}
		cout << "view_space "<< views.size() << " getted form octomap with executed time " << clock() - now_time << " ms." << endl;
	}

	View_Space(shared_ptr<Share_Data>& _share_data) {
		share_data = _share_data;
		num_of_views = share_data->num_of_views;
		now_camera_pose_world = share_data->now_camera_pose_world;
		octo_model = share_data->octo_model;
		octomap_resolution = share_data->octomap_resolution;
		//获取点云BBX
		vector<Eigen::Vector3d> points;
		for (auto& ptr : share_data->cloud_ground_truth->points) {
			Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
			points.push_back(pt);
		}
		//视点生成器
		get_view_space(points);
		//show viewspace
		if (share_data->show) {	
			share_data->access_directory(share_data->save_path);
			ofstream fout(share_data->save_path + "/view_space.txt");
			for (int i = 0; i < views.size(); i++) {
				auto coord = views[i].init_pos;
				fout << coord(0) - object_center_world(0) << ' ' << coord(1) - object_center_world(1) << ' ' << coord(2) - object_center_world(2) << '\n';
			}
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("points_on_sphere"));
			viewer1->setBackgroundColor(255, 255, 255);
			//viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			double max_z = 0;
			int index_z = 0;
			for (int i = 0; i < views.size(); i++) {
				Eigen::Vector4d X(0.03, 0, 0, 1);
				Eigen::Vector4d Y(0, 0.03, 0, 1);
				Eigen::Vector4d Z(0, 0, 0.03, 1);
				Eigen::Vector4d O(0, 0, 0, 1);
				views[i].get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), object_center_world);
				//views[i].get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), object_center_world, 1);
				Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * views[i].pose.inverse()).eval();
				//cout << view_pose_world << endl;
				X = view_pose_world * X;
				Y = view_pose_world * Y;
				Z = view_pose_world * Z;
				O = view_pose_world * O;
				//viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
				//viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
				//viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
				if( i==0 || i==3 || i==1){
					viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
					viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
					viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
				}
				else {
					viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 0, 0, 0, "X" + to_string(i));
					viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 0, 0, "Y" + to_string(i));
					viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 0, "Z" + to_string(i));
				}
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "X" + to_string(i));
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Y" + to_string(i));
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Z" + to_string(i));

				if (views[i].init_pos(1) > max_z) {
					max_z = views[i].init_pos(2);
					index_z = i;
				}
			}
			cout << "z_max_index is " << index_z << endl;

			viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");

			//add_bbx_to_cloud(viewer1);
			
			////生成一个桌面边框
			//pcl::PointXYZRGB point1;
			//point1.x = -0.4;
			//point1.y = -0.4;
			//point1.z = 0;
			//pcl::PointXYZRGB point2;
			//point2.x = -0.4;
			//point2.y = 0.4;
			//point2.z = 0;
			//pcl::PointXYZRGB point3;
			//point3.x = 0.4;
			//point3.y = 0.4;
			//point3.z = 0;
			//pcl::PointXYZRGB point4;
			//point4.x = 0.4;
			//point4.y = -0.4;
			//point4.z = 0;
			//viewer1->addLine<pcl::PointXYZRGB>(point1, point2, 0.5, 0.5, 0.5, "XY0");
			//viewer1->addLine<pcl::PointXYZRGB>(point2, point3, 0.5, 0.5, 0.5, "XY1");
			//viewer1->addLine<pcl::PointXYZRGB>(point3, point4, 0.5, 0.5, 0.5, "XY2");
			//viewer1->addLine<pcl::PointXYZRGB>(point4, point1, 0.5, 0.5, 0.5, "XY3");
			//viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "XY0");
			//viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "XY1");
			//viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "XY2");
			//viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "XY3");

			//生成一个桌面
			//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_table(new pcl::PointCloud<pcl::PointXYZRGB>);
			//for (double x = -0.4; x <= 0.4; x += 0.001) {
			//	for (double y = -0.4; y <= 0.4; y += 0.001) {
			//		pcl::PointXYZRGB point;
			//		point.x = x;
			//		point.y = y;
			//		point.z = 0;
			//		point.r = 128;
			//		point.g = 128;
			//		point.b = 128;
			//		cloud_table->push_back(point);
			//	}
			//}
			//viewer1->addPointCloud<pcl::PointXYZRGB>(cloud_table, "cloud_table");

			////设置观察点
			//pcl::visualization::Camera cam;
			//viewer1->getCameraParameters(cam);
			//cam.window_size[0] = 1280;
			//cam.window_size[1] = 720;
			//viewer1->setCameraParameters(cam);
			//viewer1->setCameraPosition(0.0, 0.2, 1.0, 0, 0, 0, 0, 0, 1);

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
		//更新一下数据区数据
		share_data->object_center_world = object_center_world;
		share_data->predicted_size = predicted_size;
		//释放内存
		points.clear();
		points.shrink_to_fit();
	}

	void add_bbx_to_cloud(pcl::visualization::PCLVisualizer::Ptr& viewer) {
		double x1 = object_center_world(0) - predicted_size;
		double x2 = object_center_world(0) + predicted_size;
		double y1 = object_center_world(1) - predicted_size;
		double y2 = object_center_world(1) + predicted_size;
		double z1 = object_center_world(2) - predicted_size;
		double z2 = object_center_world(2) + predicted_size;
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y1, z1), pcl::PointXYZ(x1, y2, z1), 0, 255, 0, "cube1");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y1, z1), pcl::PointXYZ(x2, y1, z1), 0, 255, 0, "cube2");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y1, z1), pcl::PointXYZ(x1, y1, z2), 0, 255, 0, "cube3");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z2), pcl::PointXYZ(x1, y2, z2), 0, 255, 0, "cube4");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z2), pcl::PointXYZ(x2, y1, z2), 0, 255, 0, "cube5");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z2), pcl::PointXYZ(x2, y2, z1), 0, 255, 0, "cube6");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y1, z2), pcl::PointXYZ(x1, y1, z2), 0, 255, 0, "cube8");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y1, z2), pcl::PointXYZ(x2, y1, z1), 0, 255, 0, "cube9");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y2, z2), pcl::PointXYZ(x1, y1, z2), 0, 255, 0, "cube10");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y2, z2), pcl::PointXYZ(x1, y2, z1), 0, 255, 0, "cube11");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z1), pcl::PointXYZ(x1, y2, z1), 0, 255, 0, "cube12");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z1), pcl::PointXYZ(x2, y1, z1), 0, 255, 0, "cube7");
	}

	~View_Space() {
		//释放内存
		views.clear();
		views.shrink_to_fit();
		octo_model.reset();
		share_data.reset();
		//cout << "View_Space: share_data use_count is " << share_data.use_count() << endl;
		//cout << "View_Space: octo_model use_count is " << octo_model.use_count() << endl;
	}
};
