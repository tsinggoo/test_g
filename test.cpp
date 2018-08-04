#include <iostream>
#include <string>
#include <set>
#include <unordered_set>
#include <cstdlib>
#include <iomanip>
#include <queue>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>

#include <boost/format.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/concept_check.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/g2o_types_sba_api.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>

#include "DBoW3/DBoW3.h"


using namespace std;
using namespace cv;

#define VERBOSE1
//#define VERBOSE
#define OUTPUT
//#define SHOW 
#define VIZ
//#define LOAD

const double eps = 1e-7;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct TrackInfo{
	
	// number of valid images
	unsigned int N;
	
	//camera intrinsic parameters
	Mat K;
	
	//all images
	vector<cv::Mat> colorImgs, depthImgs;
	vector<vector<KeyPoint>> all_keypoints;
	vector<cv::Mat> descriptors;
	vector<Eigen::Isometry3d> poses;
	vector<vector<DMatch>> all_matches;
	
	// loop images
	vector<int> preRealIndices;
	vector<int> curRealIndices;
	vector<Mat> preLoopColorImgs;
	vector<Mat> preLoopDepthImgs;
	vector<Mat> curLoopColorImgs;
	vector<Mat> curLoopDepthImgs;
	
	// 3D mappoints
	// PointT 格式的mappoints是带颜色的，颜色可以作为后面判断两帧匹配到的特征点之间是否是正确匹配
	PointCloudT::Ptr pointCloud;
	//store every frame mappoints 和　二维特征点一一对应
	vector<map<int, int>> all_pt2d_pt3d; // <feature id, 3d point id>
	
	//cur loop poses
	vector<Eigen::Isometry3d> curPoses;
	
	//map<point3d_id, vector<pair<keyframe_id, keypoint_id>>>
	map<int, vector<pair<int, int>>> point3d_keyframe_keypoint;
	
	//DBoW3
	vector<DBoW3::BowVector> bowVecs;
	vector<DBoW3::FeatureVector> featVecs;
	
	//after optimization
	vector<Eigen::Isometry3d> new_poses;
	PointCloudT::Ptr new_pointCloud;
	
	
	TrackInfo(){}
} trackInfo;

/*TUM2
 * 
const double cx = 325.141442;
const double cy = 249.701764;
const double fx = 520.908620;
const double fy = 521.007327;
const double depthScale = 5208.0;
*/

//TUM3
const double cx = 320.1;
const double cy = 247.6;
const double fx = 535.4;
const double fy = 539.2;
const double depthScale = 5000.0;

// fitting the key of pair for unordered_map, define like: unordered_map<pair<int, int>, bool, pairhash> loc;
struct pairhash	{ 
	template<class T1, class T2>    
	size_t operator() (const pair<T1, T2> &x) const    
	{ 
		hash<T1> h1;        
		hash<T2> h2;        
		return h1(x.first) ^ h2(x.second); 
	} 
};

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
	const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void buildMap() {
	vector<Eigen::Isometry3d>& poses = trackInfo.poses;
	vector<Mat>& colorImgs = trackInfo.colorImgs;
	vector<Mat>& depthImgs = trackInfo.depthImgs;
	
	PointCloudT::Ptr cloud(new PointCloudT);
	for(int i = 0; i < trackInfo.N; i++) {
		
		Mat& color = colorImgs[i];
		Mat& depth = depthImgs[i];
		Eigen::Isometry3d& T = poses[i];
		
		for(int v = 0; v < color.rows; v++) {
			for(int u = 0; u < color.cols; u++) {
				
				unsigned int d = depth.ptr<unsigned short>(v)[u];
				if(d == 0) continue;
				if(v%3 || u%3) continue;
				
				Eigen::Vector3d point;
				point[2] = double(d) / depthScale;
				if(point[2] > 3.0) continue;
				
				point[0] = (u-cx) * point[2] / fx;
				point[1] = (v-cy) * point[2] / fy;
				Eigen::Vector3d pointWorld = T * point;
				
				PointT p;
				p.x = pointWorld[0];
				p.y = pointWorld[1];
				p.z = pointWorld[2];
				p.b = color.data[v*color.step + u*color.channels()]; //这里注意颜色的顺序！！！
				p.g = color.data[v*color.step + u*color.channels() + 1];
				p.r = color.data[v*color.step + u*color.channels() + 2];
				
				cloud->points.push_back(p);
			}
		}
		
	}
	
	cloud->is_dense = false;
	
#ifdef VIZ1
	// for visualization
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer"));
	viewer->addPointCloud(cloud, "Map");
	viewer->spin();
	
	while(!viewer->wasStopped()) {
		viewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
#endif
	
}

void getVecDesc(Mat& desc, vector<Mat>& vDesc) {
	vDesc.clear();
	for(size_t i = 0; i < desc.rows; i++) {
		vDesc.push_back(desc.row(i));
	}
}

// matching between current Frame and reference KeyFrame
void matchByBoW(Mat& descriptors_1, Mat& descriptors_2, 
				DBoW3::FeatureVector& featVec_1, DBoW3::FeatureVector& featVec_2, 
				vector<DMatch>& matches) { //TODO
	
	unordered_set<unsigned int> markIdx; // avoid  rematch
	
	DBoW3::FeatureVector::const_iterator it_1 = featVec_1.begin();
	DBoW3::FeatureVector::const_iterator end_1 = featVec_1.end();
	DBoW3::FeatureVector::const_iterator it_2 = featVec_2.begin();
	DBoW3::FeatureVector::const_iterator end_2 = featVec_2.end();
	
	while(it_1 != end_1 && it_2 != end_2) {
		
		//步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
		//FeatureVector定义为std::map，map的元素为<node_id, std::vector<feature_id>>，feature_id是所在帧的一个特征点的编号
		if(it_1->first == it_2->first) {
			const vector<unsigned int> indices_1 = it_1->second;
			const vector<unsigned int> indices_2 = it_2->second;
			
#ifdef VERBOSE
			cout << "vector<feature>1.size() : " << indices_1.size() << endl;
			cout << "vector<feature>2.size() : " << indices_2.size() << endl;
			cout << endl;
#endif
			
			// 步骤2：遍历KF中属于该node的特征点
			for(size_t i = 0; i < indices_1.size(); i++) {
				const unsigned int realIdx_1 = indices_1[i];
				
				const cv::Mat &curDesc_1 = descriptors_1.row(realIdx_1);
				//cout << "curDesc_1: " << curDesc_1 << endl;
				
				int bestDist1 = 256; // 最好的距离（最小距离），距离越小越好
				int bestIdx = -1;
				int bestDist2 = 256; // 倒数第二好距离（倒数第二小距离）
				
				// 步骤3：遍历F中属于该node的特征点，找到了最佳匹配点
				for(size_t j = 0; j < indices_2.size(); j++) {
					const unsigned int realIdx_2 = indices_2[j];
					if(markIdx.count(realIdx_2)) continue; // 表明这个点已经被匹配过了，不再匹配，加快速度
					
					const cv::Mat &curDesc_2 = descriptors_2.row(realIdx_2);
					//cout << "curDesc_2: " << curDesc_2 << endl;
					
					const int dist = DescriptorDistance(curDesc_1, curDesc_2);
					
					if(dist < bestDist1) {
						bestDist2 = bestDist1;
						bestDist1 = dist;
						bestIdx = realIdx_2;
					} else if(dist < bestDist2) {
						bestDist2 = dist;
					}
				}
				
				// 步骤4：根据阈值剔除误匹配
				if(bestDist1 <= 40) {
					
					// 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
					if(static_cast<float>(bestDist1) < 0.7*static_cast<float>(bestDist2)) {
						cv::DMatch tmpMatch;
						tmpMatch.queryIdx = realIdx_1; // KF
						tmpMatch.trainIdx = bestIdx;  // F
						tmpMatch.distance = bestDist1;
						
						matches.push_back(tmpMatch);
						
						markIdx.insert(indices_2[bestIdx]); // set marker
					}
				}
				
			}
			it_1++;
			it_2++;
			
		} else if(it_1->first < it_2->first) {
			it_1 = featVec_1.lower_bound(it_2->first);
		} else {
			it_2 = featVec_2.lower_bound(it_1->first);
		}
	}
	
	//sort matched points pair by distance
	auto cmp = [](cv::DMatch a, cv::DMatch b) {return a.distance < b.distance; };
	sort(matches.begin(), matches.end(), cmp);
	
#ifdef VERBOSE
	cout << "matches.size() : " << matches.size() << endl;
	for(size_t i = 0; i < matches.size(); i++) {
		cout << "(" << matches[i].queryIdx << ", " << matches[i].trainIdx << "); ";
	}
	cout << endl;
#endif
	
}

void matchByBFMatcher(Mat& descs_1, Mat& descs_2, vector<DMatch>& matches) {
	
	BFMatcher matcher;
	matcher.match(descs_1, descs_2, matches);
	//matcher.knnMatch(descs_1, descs_2, matches, 2);
}

void matchByFlannMatcher(Mat& descs_1, Mat& descs_2, vector<DMatch>& matches) { // error 
	
	FlannBasedMatcher matcher;
	matcher.match(descs_1, descs_2, matches);
	//matcher.knnMatch(descs_1, descs_2, matches, 2);
}

void poseEstimation2d2d(vector<KeyPoint>& keypoints_1, 
						vector<KeyPoint>& keypoints_2,
						vector<DMatch>& matches, 
						Mat& R, Mat& t) {
			
	Mat K = ( Mat_<double> ( 3,3 ) << fx, 0, cx, 0, fy, cy, 0, 0, 1 );
	
	//-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }
    
    //-- 计算本质矩阵
    Point2d principal_point ( cx, cy );				//相机主点, TUM dataset标定值
    int focal_length = fx;						//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
#ifdef VERBOSE
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;
#endif
	
	//-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
#ifdef VERBOSE
	cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl << endl;
#endif
	
}

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}

void triangulation(const KeyPoint& kp1, 
				   const KeyPoint& kp2, 
				   const Mat& R, 
				   const Mat& t,
                   Point3d& pt_3d) {
	
	Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
	
	Mat K = ( Mat_<double> ( 3,3 ) << fx, 0, cx, 0, fy, cy, 0, 0, 1 );
	
	Point2f pt2f_1 = kp1.pt;
	Point2f pt2f_2 = kp2.pt;
	vector<Point2f> pts2f_1;
	vector<Point2f> pts2f_2;
	pts2f_1.push_back( pixel2cam(pt2f_1, K) );
	pts2f_2.push_back( pixel2cam(pt2f_2, K) );
	
	Mat pt_4d;
	triangulatePoints(T1, T2, pts2f_1, pts2f_2, pt_4d);
	
	pt_4d /= pt_4d.at<float>(3, 0); // 归一化
	pt_3d.x = pt_4d.at<float>(0, 0);
	pt_3d.y = pt_4d.at<float>(1, 0);
	pt_3d.z = pt_4d.at<float>(2, 0);
	
}

PointT getWorldPoint(Point3d& pt_3d, Eigen::Isometry3d& pose, Mat& img, int& u, int& v) {
	
	//! problem : the scale of the pt_3d　此点的单位是什么？？？
	Eigen::Vector3d eigen_point;
	eigen_point[0] = pt_3d.x;
	eigen_point[1] = pt_3d.y;
	eigen_point[2] = pt_3d.z;
	Eigen::Vector3d pointWorld = pose * eigen_point;
	
	PointT new_point;
	new_point.x = eigen_point[0];
	new_point.y = eigen_point[1];
	new_point.z = eigen_point[2];
	new_point.b = img.data[v*img.step + u*img.channels()];
	new_point.g = img.data[v*img.step + u*img.channels() + 1];
	new_point.r = img.data[v*img.step + u*img.channels() + 2];
	
	return new_point;
}

void colorMatchFilter(vector<DMatch>& matches, vector<DMatch>& colorMatches, 
					  Mat& img1, Mat& img2, 
					  vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2) {
	
	// color filter
	const int dirN = 25;
    const int dir[dirN][2] = {
		{0, 0}, 
		{1, 0}, {0, 1}, {-1, 0}, {0, -1}, 
		{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
		{2, 0},{0, 2}, {0, -2}, {-2, 0},
		{2, 1}, {1, 2}, {-2, 1}, {2, -1}, {1, -2}, {-1, 2}, {-1, -2}, {-2, -1},
		{2, 2}, {-2, 2}, {2, -2}, {-2, -2}
	};
	
	for(int i = 0; i < matches.size(); i++) {
		int k1 = matches[i].queryIdx;
		int k2 = matches[i].trainIdx;
		
		int r1 = 0, g1 = 0, b1 = 0;
		int r2 = 0, g2 = 0, b2 = 0;

		int u1 = kpts1[k1].pt.x;
		int v1 = kpts1[k1].pt.y;
		int u2 = kpts2[k2].pt.x;
		int v2 = kpts2[k2].pt.y;
		
		for(int i = 0; i < dirN; i++) {
			int uu1 = u1 + dir[i][0];
			int vv1 = v1 + dir[i][1];
			int uu2 = u2 + dir[i][0];
			int vv2 = v2 + dir[i][1];
			if(uu1 < 0 || uu1 >= img1.cols) uu1 = u1;
			if(vv1 < 0 || vv1 >= img1.rows) vv1 = v1;
			if(uu2 < 0 || uu2 >= img2.cols) uu2 = u2;
			if(vv2 < 0 || vv2 >= img2.rows) vv2 = v2;
			
			b1 += img1.data[vv1*img1.step + uu1*img1.channels()];
			g1 += img1.data[vv1*img1.step + uu1*img1.channels() + 1];
			r1 += img1.data[vv1*img1.step + uu1*img1.channels() + 2];
			b2 += img2.data[vv2*img2.step + uu2*img2.channels()];
			g2 += img2.data[vv2*img2.step + uu2*img2.channels() + 1];
			r2 += img2.data[vv2*img2.step + uu2*img2.channels() + 2];
		}
		
		r1 /= dirN;
		g1 /= dirN;
		b1 /= dirN;
		r2 /= dirN;
		g2 /= dirN;
		b2 /= dirN;
		
		int dis_r = abs(r1 - r2);
		int dis_g = abs(g1 - g2);
		int dis_b = abs(b1 - b2);
		int dis = dis_r + dis_g + dis_b;
		if(dis_r < 15 && dis_g < 15 && dis_b < 15) colorMatches.push_back(matches[i]);
		//if(dis < 35) colorMatches.push_back(matches[i]); 
		
	}
	
#ifdef VERBOSE
	cout << "color matches : " << colorMatches.size() << endl << endl;
#endif
	
}

bool colorPatchJudge(Mat& img1, Mat& img2, KeyPoint& kp1, KeyPoint& kp2) {
	
	// color filter
	const int dirN = 25;
    const int dir[dirN][2] = {
		{0, 0}, 
		{1, 0}, {0, 1}, {-1, 0}, {0, -1}, 
		{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
		{2, 0},{0, 2}, {0, -2}, {-2, 0},
		{2, 1}, {1, 2}, {-2, 1}, {2, -1}, {1, -2}, {-1, 2}, {-1, -2}, {-2, -1},
		{2, 2}, {-2, 2}, {2, -2}, {-2, -2}
	};
	
	int r1 = 0, g1 = 0, b1 = 0;
	int r2 = 0, g2 = 0, b2 = 0;
	
	int u1 = kp1.pt.x;
	int v1 = kp1.pt.y;
	int u2 = kp2.pt.x;
	int v2 = kp2.pt.y;
	
	
	for(int i = 0; i < dirN; i++) {
		int uu1 = u1 + dir[i][0];
		int vv1 = v1 + dir[i][1];
		int uu2 = u2 + dir[i][0];
		int vv2 = v2 + dir[i][1];
		if(uu1 < 0 || uu1 >= img1.cols) uu1 = u1;
		if(vv1 < 0 || vv1 >= img1.rows) vv1 = v1;
		if(uu2 < 0 || uu2 >= img2.cols) uu2 = u2;
		if(vv2 < 0 || vv2 >= img2.rows) vv2 = v2;
			
		b1 += img1.data[vv1*img1.step + uu1*img1.channels()];
		g1 += img1.data[vv1*img1.step + uu1*img1.channels() + 1];
		r1 += img1.data[vv1*img1.step + uu1*img1.channels() + 2];
		b2 += img2.data[vv2*img2.step + uu2*img2.channels()];
		g2 += img2.data[vv2*img2.step + uu2*img2.channels() + 1];
		r2 += img2.data[vv2*img2.step + uu2*img2.channels() + 2];
	}
		
	r1 /= dirN;
	g1 /= dirN;
	b1 /= dirN;
	r2 /= dirN;
	g2 /= dirN;
	b2 /= dirN;
		
	int dis_r = abs(r1 - r2);
	int dis_g = abs(g1 - g2);
	int dis_b = abs(b1 - b2);
	int dis = dis_r + dis_g + dis_b;
	
	return dis < 30;
	
}

//对极关系筛选匹配点
void checkDistEpipolarLine(const Mat& img_1, const Mat& img_2, 
						   vector<KeyPoint>& all_kpts1, vector<KeyPoint>& all_kpts2, 
						   vector<DMatch>& matches,  vector<DMatch>& ret_matches) {
	
	vector<KeyPoint> kpts1, kpts2;
	for(DMatch& dm : matches) {
		int id1 = dm.queryIdx;
		int id2 = dm.trainIdx;
		KeyPoint kp1 = all_kpts1[id1];
		KeyPoint kp2 = all_kpts2[id2];
		kpts1.push_back(kp1);
		kpts2.push_back(kp2);
	}
	
	vector<Point2f> points1, points2;
	for(int i = 0; i < kpts1.size(); i++) {
		points1.push_back(kpts1[i].pt);
		points2.push_back(kpts2[i].pt);
	}
	
	// compute scalefactor and scalefactor^2
	// pyramid 8 levels
	vector<double> vScaleFactor(8);
	vector<double> vLevelSigma2(8);
	vScaleFactor[0] = 1.0f;
	vLevelSigma2[0] = 1.0f;
	for(int i = 1; i < 8; i++) {
		vScaleFactor[i] = vScaleFactor[i-1] * 1.2; // 1.2 scalefactor
		vLevelSigma2[i] = vScaleFactor[i] * vScaleFactor[i];
	}
	
	cv::Mat F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);  
    
	//首先根据对应点计算出两视图的基础矩阵，基础矩阵包含了两个相机的外参数关系  
    std::vector<cv::Vec<float, 3>> epilines1, epilines2;  
    cv::computeCorrespondEpilines(points1, 1, F, epilines1);//计算对应点的外极线epilines是一个三元组(a,b,c)，表示点在另一视图中对应的外极线ax+by+c=0;  
    //cv::computeCorrespondEpilines(points2, 2, F, epilines2); 
	
	//若匹配到的点比较少，不足够来计算对极线，则会出现一下问题：
	/*
	OpenCV Error: Assertion failed (F.size() == Size(3,3)) in computeCorrespondEpilines, 
	file /home/tsinggoo/opencv/opencv-3.2.0/modules/calib3d/src/fundam.cpp, line 789
	terminate called after throwing an instance of 'cv::Exception'
	what():  /home/tsinggoo/opencv/opencv-3.2.0/modules/calib3d/src/fundam.cpp:789: 
	error: (-215) F.size() == Size(3,3) in function computeCorrespondEpilines

    已放弃 (核心已转储)
	*/
	
    //将图片转换为RGB图，画图的时候外极线用彩色绘制  
    cv::Mat img1, img2;  
    if (img_1.type() == CV_8UC3)  
    {  
        img_1.copyTo(img1);  
        img_2.copyTo(img2);  
    }  
    else if (img_1.type() == CV_8UC1)  
    {  
        cvtColor(img_1, img1, COLOR_GRAY2BGR);  
        cvtColor(img_2, img2, COLOR_GRAY2BGR);  
    }  
    else  
    {  
        cout << "unknow img type\n" << endl;  
        exit(0);  
    }

#ifdef SHOW
    cv::RNG& rng = theRNG();
	const int height = max(img1.rows, img2.rows);
	const int width = img1.cols + img2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	img1.copyTo(output(Rect(0, 0, img1.cols, img1.rows)));
	img2.copyTo(output(Rect(img1.cols, 0, img2.cols, img2.rows)));
#endif
	
    for(uint i = 0; i < points2.size(); i++) {
		
		//在第２图中的对极线 ax + by + c = 0
		float a = epilines1[i][0];
		float b = epilines1[i][1];
		float c = epilines1[i][2];
	
#ifdef VERBOSE
		cout << "a b c : " << a << " " << b << " " << c << endl;
#endif
		// 计算kp1特征点到极线的距离：
		// 极线l：ax + by + c = 0
		// (u,v)到l的距离为： |au+bv+c| / sqrt(a^2+b^2)

		const float num = a*points2[i].x + b*points2[i].y + c;

		const float den = a*a+b*b;

		if(den==0)
			return ;

		const float dsqr = num*num/den;
		float factor = 0.3;
		float threshold = 3.84 * vLevelSigma2[kpts2[i].octave] * factor;
		bool is_ok = dsqr < threshold;
		if(is_ok) {
			ret_matches.push_back(matches[i]);
		}
		
#ifdef VERBOSE
		cout << "dsqr : " << dsqr << "  threshold: " << threshold << endl;
		cout << is_ok << endl;
#endif
		
#ifdef SHOW
		//可视化
		Scalar color = Scalar(rng(256), rng(256), rng(256));
		Point2f lft = kpts1[i].pt;
		Point2f rht = kpts2[i].pt + Point2f((float)img1.cols, 0.f);
		if( is_ok ) { //right match
			
			line(output, lft, rht, color, 1, LINE_AA);
			circle(output, lft, 3, color, 1, LINE_AA);
			circle(output, rht, 3, color, 1, LINE_AA);
		} else {
			Scalar black = Scalar(0, 0, 0);
			line(output, lft, rht, black, 2, LINE_AA);
			circle(output, lft, 5, black, 2, LINE_AA);
			circle(output, rht, 5, black, 2, LINE_AA);
		}
#endif
		
	}
	
#ifdef VERBOSE
	cout << "ret_matches.size() : " << ret_matches.size() << endl;
#endif
	
	
#ifdef SHOW
	cout << "output.size() : " << output.size() << endl;
	cout << "show output :" << endl;
	imshow("show", output);
	waitKey(0);
#endif
	
}


void descriptorMatchFilter(vector<DMatch>& matches, vector<DMatch>& descMatches, 
						   Mat& img1, Mat& img2, 
						   vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2) {
	
	
	float min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < matches.size(); i++ )
    {
        float dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

#ifdef VERBOSE
    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );
#endif
	
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < matches.size(); i++ )
    {
        if ( matches[i].distance <= max ( (float)1.5*min_dist, (float)25.0 ) )
        {
            descMatches.push_back ( matches[i] );
        }
    }
    
}

Point3f point3dRecomputeWithDepthScale(Point3f& point) {
	Point3f p;
	p.z = double(point.z) / depthScale;
	p.x = (point.x - cx) / fx * p.z;
	p.y = (point.y - cy) / fy * p.z;
	
	return p;
}

unsigned int getDepth(Point2f& pt2f, Mat& depthImg) {
	int u = pt2f.x; //u v 和　x y　一 一对应
	int v = pt2f.y;
			
	unsigned int d = depthImg.ptr<ushort>(v)[u]; //注意
	if(d == 0) { //黑点, 深度为０，就从旁边找替代值
		const int dir[8][2] = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		unsigned int sumD = 0;
		unsigned int cnt = 0;
		for(int k = 0; k < 8; k++) {
			int uu = u + dir[k][0];
			int vv = v + dir[k][1];
			if(uu < 0 || uu >= depthImg.cols || vv < 0 || vv >= depthImg.rows) continue;
					
			unsigned int tmpD = depthImg.ptr<ushort>(vv)[uu];
			if(tmpD == 0) continue;
					
			sumD += tmpD;
			++cnt;
		}
		if(cnt) d = sumD / cnt;
	}
	
	return d;
}

double get3dDistance(PointT& p1, PointT& p2) {
	return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

void create3dPointsByBoW(DBoW3::Vocabulary& vocab, 
						 Mat& imgColor1, Mat& imgColor2, 
						 Mat& imgDepth1, Mat& imgDepth2, 
						 vector<KeyPoint>& vkps1, vector<KeyPoint>& vkps2, 
						 Mat& descriptors1, Mat& descriptors2,
						 DBoW3::BowVector& bowVec_1, DBoW3::BowVector& bowVec_2, 
						 DBoW3::FeatureVector& featVec_1, DBoW3::FeatureVector& featVec_2, 
						 Eigen::Isometry3d& pose_1, Eigen::Isometry3d& pose_2, 
						 vector<DMatch>& matches) {
	
	// match by BoW way
	vector<DMatch> raw_matches;
	matchByBoW(descriptors1, descriptors2, featVec_1, featVec_2, raw_matches);
	
	//描述子距离筛选
	vector<DMatch> desc_matches;
	descriptorMatchFilter(raw_matches, desc_matches, imgColor1, imgColor2, vkps1, vkps2);
	
	//颜色筛选
	vector<DMatch> color_matches;
	colorMatchFilter(desc_matches, color_matches, imgColor1, imgColor2, vkps1, vkps2);
	
	//对极几何关系筛选
	vector<DMatch> epipolar_matches;
	checkDistEpipolarLine(imgColor1, imgColor2, vkps1, vkps2, color_matches, epipolar_matches);
	
	//new mappoints
	//map<int, PointT> curr_frame_mappoints; //TODO to be collected
	
	matches = epipolar_matches;
	
	// for triangulation
	Mat R, t;
	poseEstimation2d2d(vkps1, vkps2, matches, R, t);
	
	// record triangulation points for scale fixing
	vector<PointT> record_trian; //
	vector<PointT> record_depth; // <idx2, PointT>
	
	//　记录二维特征点和三维云点的对应关系
	map<int, int>& pre_pt2d_pt3d = trackInfo.all_pt2d_pt3d.back(); //前一帧的记录
	map<int, int> cur_pt2d_pt3d;                                  //当前帧的记录
	
	// 记录需要重新计算的３ｄ点的编号
	vector<int> record_pointcloud_no_depth;
	//　记录需要重新计算的３ｄ点　对应的　两帧编号
	map<int, pair<int, int>> record_2dpoint_pair;// <3d point id> , <keypoint_id of lastImg, keypoint_id of currImg>
	
	for(DMatch &dm : matches) {
		int idx1 = dm.queryIdx;
		int idx2 = dm.trainIdx;
		
		KeyPoint kp1 = vkps1[idx1];
		KeyPoint kp2 = vkps2[idx2];
		int u1 = kp1.pt.x;
		int v1 = kp1.pt.y;
		int u2 = kp2.pt.x;
		int v2 = kp2.pt.y;
		unsigned int d1 = imgDepth1.ptr<unsigned short>(v1)[u1];
		if(d1 == 0) d1 = getDepth(kp1.pt, imgDepth1);
		unsigned int d2 = imgDepth2.ptr<unsigned short>(v2)[u2];
		if(d2 == 0) d2 = getDepth(kp2.pt, imgDepth2);
		
		// 用Triangulation三角化出云点
		Point3d pt_3d;
		triangulation(kp1, kp2, R, t, pt_3d);
		PointT tri_point = getWorldPoint(pt_3d, pose_1, imgColor1, u1, v1);
		
		bool is_3dpoint_exist = false;
		if(pre_pt2d_pt3d.count(idx1)) { //　已经有对应的三维云点，不用创建了
			cur_pt2d_pt3d[idx2] = pre_pt2d_pt3d[idx1];
			is_3dpoint_exist = true;
		}//虽然这里可以直接continue跳出当前循环，但是我们还需要为恢复三角化的点的尺度而继续计算　三角化的点和ｄｅｐｔｈ得到的点的坐标,为后面恢复尺度做准备
		
		
		if(d1 == 0 && d2 == 0) {
			if(!is_3dpoint_exist) { // 说明这个点不仅不能直接深度化　而且　从来没有出现过　－＞　必须三角化，后期需处理尺度
				PointT new_point = tri_point;
				
				trackInfo.pointCloud->points.push_back(new_point);
				cur_pt2d_pt3d[idx2] = trackInfo.pointCloud->points.size()-1;
				pre_pt2d_pt3d[idx1] = cur_pt2d_pt3d[idx2];
				
				record_pointcloud_no_depth.push_back(cur_pt2d_pt3d[idx2]); // 记录下需后期处理的三维云点
			}// else 表示 : 虽然这个点在两张图像上都没有深度，但是这个点很早前就是地图三维点了 
			
		} else { // 可以同时三角化计算　和　深度化计算
			
			record_trian.push_back(tri_point);
			
			if(is_3dpoint_exist) { //对应的三维云点已经存在了，则无需创建　 
				//三维点已经存在，则直接拿过来就行了
				int id_3d = pre_pt2d_pt3d[idx1];
				PointT& tmp = trackInfo.pointCloud->points[id_3d];
				
				record_depth.push_back(tmp);
				
			} else {
				
				//三维点不存在，则深度化计算三维点
				PointT new_point;
				
				//从第一幅图像中提取深度
				if(d1 != 0) { 
					
					Eigen::Vector3d eigen_point;
					eigen_point[2] = double(d1) / depthScale;
					eigen_point[0] = (u1-cx) * eigen_point[2] / fx;
					eigen_point[1] = (v1-cy) * eigen_point[2] / fy;
					Eigen::Vector3d pointWorld = pose_1 * eigen_point;
			
					
					new_point.x = pointWorld[0];
					new_point.y = pointWorld[1];
					new_point.z = pointWorld[2];
					new_point.b = imgColor1.data[v1*imgColor1.step + u1*imgColor1.channels()];
					new_point.g = imgColor1.data[v1*imgColor1.step + u1*imgColor1.channels() + 1];
					new_point.r = imgColor1.data[v1*imgColor1.step + u1*imgColor1.channels() + 2];
					
				} else { //从第二幅图像中提取深度
					
					Eigen::Vector3d eigen_point;
					eigen_point[2] = double(d2) / depthScale;
					eigen_point[0] = (u2-cx) * eigen_point[2] / fx;
					eigen_point[1] = (v2-cy) * eigen_point[2] / fy;
					Eigen::Vector3d pointWorld = pose_2 * eigen_point;
			
					
					new_point.x = pointWorld[0];
					new_point.y = pointWorld[1];
					new_point.z = pointWorld[2];
					new_point.b = imgColor2.data[v2*imgColor2.step + u2*imgColor2.channels()];
					new_point.g = imgColor2.data[v2*imgColor2.step + u2*imgColor2.channels() + 1];
					new_point.r = imgColor2.data[v2*imgColor2.step + u2*imgColor2.channels() + 2];
				}
				
				trackInfo.pointCloud->points.push_back(new_point);
				cur_pt2d_pt3d[idx2] = trackInfo.pointCloud->points.size()-1;
				pre_pt2d_pt3d[idx1] = cur_pt2d_pt3d[idx2];
				
				record_depth.push_back(new_point);
				
			}
		}
		
		

	}
	
	//　计算三角化地图云点和深度图恢复地图云点之间的尺度因子std_factor
	vector<double> odds;
	double sumOdds = 0.;
	for(int i = 0; i < record_trian.size(); i++) {
		double tmpOdd = record_depth[i].z / record_trian[i].z;
		odds.push_back(tmpOdd);
		sumOdds += tmpOdd;
	}
	double std_factor = sumOdds / (double)odds.size();
	
	//　三角化的云点坐标的矫正

	
	//更新云点集pointCloud中三角化过还未统一尺度的云点
	for(int& ptId : record_pointcloud_no_depth) {
		trackInfo.pointCloud->points[ptId].x *= std_factor;
		trackInfo.pointCloud->points[ptId].y *= std_factor;
		trackInfo.pointCloud->points[ptId].z *= std_factor;
	}
	
	//将当前帧的pt2d_pt3d加入到all_pt2d_pt3d中
	trackInfo.all_pt2d_pt3d.push_back(cur_pt2d_pt3d);
	
}

void create3dPointsByBoW_withoutTriangulation(DBoW3::Vocabulary& vocab, 
						 Mat& imgColor1, Mat& imgColor2, 
						 Mat& imgDepth1, Mat& imgDepth2, 
						 vector<KeyPoint>& vkps1, vector<KeyPoint>& vkps2, 
						 Mat& descriptors1, Mat& descriptors2,
						 DBoW3::BowVector& bowVec_1, DBoW3::BowVector& bowVec_2, 
						 DBoW3::FeatureVector& featVec_1, DBoW3::FeatureVector& featVec_2, 
						 Eigen::Isometry3d& pose_1, Eigen::Isometry3d& pose_2, 
						 vector<DMatch>& matches) {
	
	// match by BoW way
	vector<DMatch> raw_matches;
	matchByBoW(descriptors1, descriptors2, featVec_1, featVec_2, raw_matches);
	
	//描述子距离筛选
	vector<DMatch> desc_matches;
	descriptorMatchFilter(raw_matches, desc_matches, imgColor1, imgColor2, vkps1, vkps2);
		
	//颜色筛选
	vector<DMatch> color_matches;
	colorMatchFilter(desc_matches, color_matches, imgColor1, imgColor2, vkps1, vkps2);
	
	//对极几何关系筛选
	vector<DMatch> epipolar_matches;
	checkDistEpipolarLine(imgColor1, imgColor2, vkps1, vkps2, color_matches, epipolar_matches);
	
	matches = epipolar_matches;
	
	//　记录二维特征点和三维云点的对应关系
	map<int, int>& pre_pt2d_pt3d = trackInfo.all_pt2d_pt3d.back(); //前一帧的记录
	map<int, int> cur_pt2d_pt3d;                                  //当前帧的记录
	
	
	for(DMatch &dm : matches) {
		int idx1 = dm.queryIdx;
		int idx2 = dm.trainIdx;
		
		KeyPoint kp1 = vkps1[idx1];
		KeyPoint kp2 = vkps2[idx2];
		int u1 = kp1.pt.x;
		int v1 = kp1.pt.y;
		int u2 = kp2.pt.x;
		int v2 = kp2.pt.y;
		unsigned int d1 = imgDepth1.ptr<unsigned short>(v1)[u1];
		if(d1 == 0) d1 = getDepth(kp1.pt, imgDepth1);
		unsigned int d2 = imgDepth2.ptr<unsigned short>(v2)[u2];
		if(d2 == 0) d2 = getDepth(kp2.pt, imgDepth2);
		
		
		bool is_3dpoint_exist = false;
		if(pre_pt2d_pt3d.count(idx1)) { //　已经有对应的三维云点，不用创建了
			cur_pt2d_pt3d[idx2] = pre_pt2d_pt3d[idx1];
			is_3dpoint_exist = true;
			
			continue;
		}//虽然这里可以直接continue跳出当前循环，但是我们还需要为恢复三角化的点的尺度而继续计算　三角化的点和ｄｅｐｔｈ得到的点的坐标,为后面恢复尺度做准备
		
		double dd1 = double(d1) / depthScale;
		double dd2 = double(d2) / depthScale;
		if(dd1 > 3. || dd2 > 3.) continue;
		
		if(!(d1 == 0 && d2 == 0)){ 
				
				//三维点不存在，则深度化计算三维点
			PointT new_point;
				
				//从第一幅图像中提取深度
			if(d1 != 0) { 
					
				Eigen::Vector3d eigen_point;
				eigen_point[2] = double(d1) / depthScale;
				eigen_point[0] = (u1-cx) * eigen_point[2] / fx;
				eigen_point[1] = (v1-cy) * eigen_point[2] / fy;
				Eigen::Vector3d pointWorld = pose_1 * eigen_point;
			
					
				new_point.x = pointWorld[0];
				new_point.y = pointWorld[1];
				new_point.z = pointWorld[2];
				new_point.b = imgColor1.data[v1*imgColor1.step + u1*imgColor1.channels()];
				new_point.g = imgColor1.data[v1*imgColor1.step + u1*imgColor1.channels() + 1];
				new_point.r = imgColor1.data[v1*imgColor1.step + u1*imgColor1.channels() + 2];
					
			} else { //从第二幅图像中提取深度
					
				Eigen::Vector3d eigen_point;
				eigen_point[2] = double(d2) / depthScale;
				eigen_point[0] = (u2-cx) * eigen_point[2] / fx;
				eigen_point[1] = (v2-cy) * eigen_point[2] / fy;
				Eigen::Vector3d pointWorld = pose_2 * eigen_point;
			
					
				new_point.x = pointWorld[0];
				new_point.y = pointWorld[1];
				new_point.z = pointWorld[2];
				new_point.b = imgColor2.data[v2*imgColor2.step + u2*imgColor2.channels()];
				new_point.g = imgColor2.data[v2*imgColor2.step + u2*imgColor2.channels() + 1];
				new_point.r = imgColor2.data[v2*imgColor2.step + u2*imgColor2.channels() + 2];
			}
				
			trackInfo.pointCloud->points.push_back(new_point);
			cur_pt2d_pt3d[idx2] = trackInfo.pointCloud->points.size()-1;
			pre_pt2d_pt3d[idx1] = cur_pt2d_pt3d[idx2];
				
				
		}
		
		

	}
	
	//将当前帧的pt2d_pt3d加入到all_pt2d_pt3d中
	trackInfo.all_pt2d_pt3d.push_back(cur_pt2d_pt3d);
	
}

// get first KF mappoints for initialization
void initializeMap(vector<KeyPoint>& first_keypoints, Mat& color_image, Mat& depth_image) {
	
	//map<int, PointT> tmpPointCloud;
	map<int, int> pt2d_pt3d;
	Eigen::Isometry3d pose = trackInfo.poses.front();
	
	for(int i = 0; i < first_keypoints.size(); i++) {
		
		Point2f p2f = first_keypoints[i].pt;
		int u = p2f.x;
		int v = p2f.y;
		
		PointT curPoint;
		unsigned int d = depth_image.ptr<unsigned short>(v)[u];
		//cout << "getDepth " << i << endl;
		if(d == 0) d = getDepth(p2f, depth_image);
		if(d == 0) {
			//
			continue;
		}
		
		Eigen::Vector3d point;
		point[2] = double(d) / depthScale;
		if(point[2] > 3.5) continue;
		point[0] = (u-cx) * point[2] / fx;
		point[1] = (v-cy) * point[2] / fy;
		Eigen::Vector3d pointWorld = pose * point;
		
		
		// get map point with depthScale 
		curPoint.x = pointWorld[0];
		curPoint.y = pointWorld[1];
		curPoint.z = pointWorld[2];
		curPoint.b = color_image.data[v*color_image.step + u*color_image.channels()];
		curPoint.g = color_image.data[v*color_image.step + u*color_image.channels() + 1];
		curPoint.r = color_image.data[v*color_image.step + u*color_image.channels() + 2];
		
		//tmpPointCloud.insert(make_pair(i, curPoint));
		trackInfo.pointCloud->points.push_back(curPoint);
		pt2d_pt3d[i] = trackInfo.pointCloud->points.size()-1;
	}
	
	trackInfo.all_pt2d_pt3d.push_back(pt2d_pt3d);
}

void trackByBoW() {
	
	//DBoW3 Vocabulary
	cout << "Loading ORB Vocabulary..." << endl;
    DBoW3::Vocabulary vocab("/home/tsinggoo/gu_RGBDSLAM/Vocabulary/orbvoc.dbow3");
    if(vocab.empty()) {
		cerr << "Vocabulary does not exist." << endl;
		return ;
	}
	cout <<  "Loading Vocabulary Done." << endl << endl;
	
	unsigned int N = trackInfo.N;
	
	vector<int>& preRealIndices   = trackInfo.preRealIndices;
	vector<int>& curRealIndices   = trackInfo.curRealIndices;
	vector<Mat>& preLoopColorImgs = trackInfo.preLoopColorImgs;
	vector<Mat>& preLoopDepthImgs = trackInfo.preLoopDepthImgs;
	vector<Mat>& curLoopColorImgs = trackInfo.curLoopColorImgs;
	vector<Mat>& curLoopDepthImgs = trackInfo.curLoopDepthImgs;
	
	vector<Mat>& colorImgs                  = trackInfo.colorImgs;
	vector<Mat>& depthImgs                  = trackInfo.depthImgs;
	vector<vector<KeyPoint>>& all_keypoints = trackInfo.all_keypoints;
	vector<Mat>& descriptors                = trackInfo.descriptors;
	vector<Eigen::Isometry3d>& poses        = trackInfo.poses;
	
	//compute bowVector and featureVector for each image
	
	cout << "Computing bowVector and featureVector for each image ..." << endl;
	vector<DBoW3::BowVector>& bowVecs = trackInfo.bowVecs;
	vector<DBoW3::FeatureVector>& featVecs = trackInfo.featVecs;
	for(int i = 0; i < N; i++) {
		DBoW3::BowVector bv;
		DBoW3::FeatureVector fv;
		vector<Mat> vDesc;
		getVecDesc(descriptors[i], vDesc);
		vocab.transform(vDesc, bv, fv, 5); // 4 less match
		
		bowVecs.push_back(bv);
		featVecs.push_back(fv);
	}
	cout << "Done." << endl << endl;
	
	// get first frame mappoints
	cout << "initializeMap ..." << endl;
	initializeMap(all_keypoints.front(), colorImgs.front(), depthImgs.front());
	cout << "Done." << endl << endl;
	
	vector<vector<DMatch>>& all_matches = trackInfo.all_matches;
	cout << "create 3d points by BoW ..." << endl;
	for(int i = 1; i < N; i++) {
		int lastId = i-1, currId = i;
		
		//cout << "image" << i << " ..." << endl;
		
		vector<DMatch> matches;
		create3dPointsByBoW_withoutTriangulation(vocab, colorImgs[lastId], colorImgs[currId], depthImgs[lastId], depthImgs[currId], 
							all_keypoints[lastId], all_keypoints[currId], descriptors[lastId], descriptors[currId],
 							bowVecs[lastId], bowVecs[currId], featVecs[lastId], featVecs[currId], 
							poses[lastId], poses[currId], matches); 
		
		all_matches.push_back(matches);
		
#ifdef VERBOSE1
			cout << "image:" << i-1 << " match image:" << i << " matches.size() : " << matches.size() << endl;
#endif
		
	}
	cout << "Done." << endl << endl;
	
	cout << "poses.size() : " << poses.size() << endl;
	cout << "pointcloud.size() : " << trackInfo.pointCloud->points.size() << endl << endl;

#ifdef VIZ1
	// for visualization
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer_before"));
	viewer->addPointCloud(trackInfo.pointCloud, "Map_before");
	viewer->spin();
	
	while(!viewer->wasStopped()) {
		viewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
#endif
	
}

//cv::Mat to Eigen
Eigen::Isometry3d cvMat2Eigen(Mat& rvec, Mat& tvec) {
	
	Mat R;
	Rodrigues(rvec, R);
	Eigen::Matrix3d r;
	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			r(i, j) = R.at<double>(i, j);
		}
	}
	
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	
	Eigen::AngleAxisd angle(r);
	T = angle;
	T(0, 3) = tvec.at<double>(0, 0);
	T(1, 3) = tvec.at<double>(1, 0);
	T(2, 3) = tvec.at<double>(2, 0);
	
	return T;
	
}

Eigen::Isometry3d cvMat2Eigen(Mat& t) {
	
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	for(int i = 0; i < t.rows; i++) {
		for(int j = 0; j < t.cols; j++) {
			T(i, j) = t.at<double>(i,j);
		}
	}
	
	return T;
}

//Tcw to Twc
Mat Tcw2Twc(const Mat &Tcw) {
	
	Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
	Mat tcw = Tcw.rowRange(0, 3).col(3);
	Mat Rwc = Rcw.t(); // 前提是Rcw为正交矩阵,正交矩阵的逆等于转置
	Mat Ow = -Rwc * tcw;
	
#ifdef VERBOSE
	cout << "Rwc: " << endl << Rwc << endl; 
	double r00 = Rwc.at<double>(0, 0);
	double r01 = Rwc.at<double>(0, 1);
	double r02 = Rwc.at<double>(0, 2);
	double r10 = Rwc.at<double>(1, 0);
	double r11 = Rwc.at<double>(1, 1);
	double r12 = Rwc.at<double>(1, 2);
	double r20 = Rwc.at<double>(2, 0);
	double r21 = Rwc.at<double>(2, 1);
	double r22 = Rwc.at<double>(2, 2);
	double v = r00*(r11*r22-r12*r21) - r01*(r10*r22-r20*r12) + r02*(r10*r21-r11*r20);
	cout << "秩 : " << v << endl;
#endif
	
	Mat Twc = Mat::eye(4, 4, Tcw.type());
	Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
	Ow.copyTo(Twc.rowRange(0, 3).col(3));
	
	return Twc;
}

Mat Tcw2Twc(const Eigen::Isometry3d& Tcw) {
	Mat T_cw = Mat::eye(4, 4, CV_64F);
	for(int i = 0; i < Tcw.matrix().rows(); i++) {
		for(int j = 0; j < Tcw.matrix().cols(); j++) {
			T_cw.at<double>(i,j) = Tcw.matrix()(i, j);
		}
	}
	
	return Tcw2Twc(T_cw);
}


void loopPoseEstimationPnP() {
	
	vector<Mat>& preLoopColorImgs = trackInfo.preLoopColorImgs;
	vector<Mat>& curLoopColorImgs = trackInfo.curLoopColorImgs;
	vector<Mat>& preLoopDepthImgs = trackInfo.preLoopDepthImgs;
	vector<Mat>& curLoopDepthImgs = trackInfo.curLoopDepthImgs;
	vector<int>& preRealIndices = trackInfo.preRealIndices;
	vector<int>& curRealIndices = trackInfo.curRealIndices;
	vector<vector<KeyPoint>>& all_keypoints = trackInfo.all_keypoints;
	vector<Mat>& all_descriptors = trackInfo.descriptors;
	
	vector<Eigen::Isometry3d>& poses = trackInfo.poses;
	//vector<map<int, PointT>>& trackedPointCloud = trackInfo.trackedPointCloud;
	
	vector<DBoW3::BowVector>& bowVecs = trackInfo.bowVecs;
	vector<DBoW3::FeatureVector>& featVecs = trackInfo.featVecs;
	
	vector<map<int, int>>& all_pt2d_pt3d = trackInfo.all_pt2d_pt3d;
	
	//picture 2d keypoints
	vector<vector<Point2f>> pre_points2d;
	vector<vector<Point2f>> cur_points2d;
	for(int& idx : preRealIndices) {
		vector<Point2f> tmp_points2d;
		for(KeyPoint& kp : all_keypoints[idx]) {
			tmp_points2d.push_back(kp.pt);
		}
		pre_points2d.push_back(tmp_points2d);
	}
	for(int& idx : curRealIndices) {
		vector<Point2f> tmp_points2d;
		for(KeyPoint& kp : all_keypoints[idx]) {
			tmp_points2d.push_back(kp.pt);
		}
		cur_points2d.push_back(tmp_points2d);
	}
	
	// create pre loop 3d points
	vector<vector<Point3f>> pre_points3d;
	vector<unordered_set<int>> depth_zero_mark;
	for(int i = 0; i < preRealIndices.size(); i++) {
		int idx = preRealIndices[i];
		vector<Point3f> tmp_points3d;
		unordered_set<int> zero_mark;
		Mat& depth = preLoopDepthImgs[i];
		Eigen::Isometry3d T = poses[idx];
		
		for(int j = 0; j < pre_points2d[i].size(); j++) {
			int u = pre_points2d[i][j].x;
			int v = pre_points2d[i][j].y;
			
			unsigned int d = depth.ptr<unsigned short>(v)[u];
			if(d == 0) {
				zero_mark.insert(j);
				tmp_points3d.push_back(Point3f(0, 0, 0));
				continue;
			}
			
			Eigen::Vector3d point;
			point[2] = double(d) / depthScale;
			point[0] = (u-cx) * point[2] / fx;
			point[1] = (v-cy) * point[2] / fy;
			Eigen::Vector3d point_world =  T * point;
			
			tmp_points3d.push_back(Point3f(point_world[0], point_world[1], point_world[2]));
			
		}
		
		pre_points3d.push_back(tmp_points3d);
		depth_zero_mark.push_back(zero_mark);
	}
	
	
	//匹配：cur loop 分别和　pre loop　的三个帧匹配然后ｐｎｐ，最后选取均值作为curloop的位姿
	vector<Eigen::Isometry3d>& curPoses = trackInfo.curPoses; //当前ｌｏｏｐ帧的位姿
	
	for(int i = 0; i < curRealIndices.size(); i++) {
		int curId = curRealIndices[i];
		vector<Eigen::Isometry3d> three_Twc;
		for(int j = 0; j < preRealIndices.size(); j++) {
			int preId = preRealIndices[j];
			
			Mat& descriptors1 = all_descriptors[preId];
			Mat& descriptors2 = all_descriptors[curId];
			DBoW3::FeatureVector& featVec_1 = featVecs[preId];
			DBoW3::FeatureVector& featVec_2 = featVecs[curId];
			Mat& imgColor1 = preLoopColorImgs[j];
			Mat& imgColor2 = curLoopColorImgs[i];
			vector<KeyPoint>& vkps1 = all_keypoints[preId];
			vector<KeyPoint>& vkps2 = all_keypoints[curId];
			
			// match by BoW way
			vector<DMatch> raw_matches;
			matchByBoW(descriptors1, descriptors2, featVec_1, featVec_2, raw_matches);
	
			//描述子距离筛选
			vector<DMatch> desc_matches;
			descriptorMatchFilter(raw_matches, desc_matches, imgColor1, imgColor2, vkps1, vkps2);
	
			//颜色筛选
			vector<DMatch> color_matches;
			colorMatchFilter(desc_matches, color_matches, imgColor1, imgColor2, vkps1, vkps2);
	
			//对极几何关系筛选
			vector<DMatch> epipolar_matches;
			checkDistEpipolarLine(imgColor1, imgColor2, vkps1, vkps2, color_matches, epipolar_matches);
			
			vector<DMatch>& matches = epipolar_matches;
			
			// get corresponding 2d points & 3d points
			vector<Point3f> pts_obj;
			vector<Point2f> pts_img;
			unordered_set<int>& curMark = depth_zero_mark[j];
			vector<Point3f>& raw_pts_obj = pre_points3d[j];
			vector<Point2f>& raw_pts_img = cur_points2d[i];
			for(DMatch& dm : matches) {
				int id1 = dm.queryIdx;
				int id2 = dm.trainIdx;
				
				//第一副图中没深度的关键点不能构建三维地图点
				if(curMark.count(id1)) continue;
				
				pts_obj.push_back(raw_pts_obj[id1]);
				pts_img.push_back(raw_pts_img[id2]);
				
			}
			
#ifdef VERBOSE
			cout << "preImg : " << preId << "  curImg : " << curId << endl;
			cout << "pts_obj.size() : " << pts_obj.size() << endl << endl;
#endif
			
			
			//compute R t by PnP
#ifdef VERBOSE
			cout << "Solving PnP ..." << endl;
#endif
			
			Mat& cameraMatrix = trackInfo.K;
			Mat rvec, tvec, inliers;
			solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers); // rcw & tcw
			
#ifdef VERBOSE
			cout << "inliers.size() : " << inliers.rows << endl << endl;
#endif
			// get Tcw
			Eigen::Isometry3d T = cvMat2Eigen(rvec, tvec);
#ifdef VERBOSE
			cout << "T : " << T.matrix() << endl << endl;
#endif
			
			Mat T_wc = Tcw2Twc(T);
			Eigen::Isometry3d Twc = cvMat2Eigen(T_wc); //!!!这个值很有可能是出错的，有待后面考证
			
			three_Twc.push_back(Twc);
			
		}
		
		
		// get average pose
		Eigen::Isometry3d sumT;
		sumT.matrix() = Eigen::Matrix4d::Zero();
		for(auto& t : three_Twc) {
			for(int i = 0; i < sumT.matrix().rows(); i++) {
				for(int j = 0; j < sumT.matrix().cols(); j++) {
					sumT(i, j) += t(i, j); 
				}
			}
		}
		
		sumT.matrix() /= 3.;
		curPoses.push_back(sumT);
		
	}
	
#ifdef VIZ1
	vector<Eigen::Isometry3d> Poses;
	unordered_set<int> mark;
	cout << "curRealIndices.size() : " << curRealIndices.size() << endl;
	for(int i : curRealIndices) {
		mark.insert(i);
		cout << "real id : " << i << endl;
	}
	
	cout << "pose before pnp vs after pnp ... " << endl << endl; 
	int id = 0;
	for(int i = 0; i < trackInfo.poses.size(); i++) {
		if(mark.count(i)){
			Poses.push_back(curPoses[id++]);
			cout << "pose real id : " << i << endl;
			cout << "pose before pnp: " << endl << trackInfo.poses[i].matrix() << endl;
			cout << "pose after  pnp: " << endl << curPoses[id-1].matrix() << endl << endl; 
		} else {
			Poses.push_back(trackInfo.poses[i]);
		}
	}
	cout << "pose compare Done." << endl << endl;
	
	vector<vector<DMatch>>& all_matches = trackInfo.all_matches;
	cout << "images.size() :      " << trackInfo.colorImgs.size() << endl;
	cout << "all_matches.size() : " << all_matches.size() << endl << endl;
	vector<Mat>& colorImgs = trackInfo.colorImgs;
	vector<Mat>& depthImgs = trackInfo.depthImgs;
	// build map
	PointCloudT::Ptr cloud(new PointCloudT);
	for(int i = 0; i < all_matches.size(); i++) {
		int id1 = i;
		int id2 = i+1;
		//if(!mark.count(id2)) continue;
		if(id1 == 0) {
			Mat& color1 = colorImgs.front();
			Mat& depth1 = depthImgs.front();
			Eigen::Isometry3d& T1 = Poses.front();
			
			vector<KeyPoint>& cur_kps = all_keypoints.front();
			for(int j = 0; j < cur_kps.size(); j++) {
				int u1 = cur_kps[j].pt.x;
				int v1 = cur_kps[j].pt.y;
 				
				unsigned int d1 = depth1.ptr<unsigned short>(v1)[u1];
				if(d1 == 0) continue;
				
				Eigen::Vector3d point1;
				point1[2] = double(d1) / depthScale;
				if(point1[2] > 3.0) continue;
				
				point1[0] = (u1-cx) * point1[2] / fx;
				point1[1] = (v1-cy) * point1[2] / fy;
				Eigen::Vector3d pointWorld1 = T1 * point1;
				
				PointT p1;
				p1.x = pointWorld1[0];
				p1.y = pointWorld1[1];
				p1.z = pointWorld1[2];
				p1.b = color1.data[v1*color1.step + u1*color1.channels()];
				p1.g = color1.data[v1*color1.step + u1*color1.channels() + 1];
				p1.r = color1.data[v1*color1.step + u1*color1.channels() + 2];
				
				cloud->points.push_back(p1);
			}
		}
		
		
		Mat& color = colorImgs[id2];
		Mat& depth = depthImgs[id2];
		Eigen::Isometry3d& T = Poses[id2];
		
		vector<KeyPoint>& cur_kps = all_keypoints[id2];
		for(KeyPoint& kp : cur_kps) {
			int u = kp.pt.x;
			int v = kp.pt.y;
			
			unsigned int d = depth.ptr<unsigned short>(v)[u];
			if(d == 0) continue;
			
			Eigen::Vector3d point;
			point[2] = double(d) / depthScale;
			if(point[2] > 3.0) continue;
			
			point[0] = (u-cx) * point[2] / fx;
			point[1] = (v-cy) * point[2] / fy;
			Eigen::Vector3d pointWorld = T * point;
			
			PointT p;
			p.x = pointWorld[0];
			p.y = pointWorld[1];
			p.z = pointWorld[2];
			p.b = color.data[v*color.step + u*color.channels()];
			p.g = color.data[v*color.step + u*color.channels() + 1];
			p.r = color.data[v*color.step + u*color.channels() + 2];
			
			cloud->points.push_back(p);
		}
		
	}
	
	cloud->is_dense = false;
	// for visualization
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer_after_pnp"));
	viewer->addPointCloud(cloud, "Map_after_pnp");
	viewer->spin();
	
	while(!viewer->wasStopped()) {
		viewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	
#endif
	
	
}

g2o::SE3Quat Eigen2SE3Quat(Eigen::Isometry3d& T) {
	
	Eigen::Matrix<double, 3, 3> R;
	R << T(0, 0), T(0, 1), T(0, 2), 
		 T(1, 0), T(1, 1), T(1, 2),
		 T(2, 0), T(2, 1), T(2, 2);
	
	Eigen::Matrix<double, 3, 1> t(T(0, 3), T(1, 3), T(2, 3));
	
	return g2o::SE3Quat(R, t);
}

Eigen::Vector3d PointT2Vector3d(PointT& point) {
	return Eigen::Vector3d(point.x, point.y, point.z);
}

Eigen::Isometry3d toIsometry(const g2o::SE3Quat& SE3) {
	Eigen::Isometry3d T;
	Eigen::Matrix4d eigMat = SE3.to_homogeneous_matrix();
	T.matrix() = eigMat;
	
	return T;
}

PointT toPointT(const Eigen::Matrix<double, 3, 1>& vPoint) {
	PointT pt;
	pt.x = vPoint(0, 0);
	pt.y = vPoint(1, 0);
	pt.z = vPoint(2, 0);
	
	return pt;
}

void getPoint3d2KeyframeAndKeypoint() {
	
	vector<map<int, int>>& all_pt2d_pt3d = trackInfo.all_pt2d_pt3d;
	map<int, vector<pair<int, int>>>& p3d_kf_kp = trackInfo.point3d_keyframe_keypoint;
	
	for(int i = 0; i < all_pt2d_pt3d.size(); i++) {
		//cout << endl << "cur image : " << i << endl;
		for(auto& m : all_pt2d_pt3d[i]) {
			int id_3d = m.second;
			int id_2d = m.first;
			
			p3d_kf_kp[id_3d].push_back(make_pair(i, id_2d));
			//cout << "(kf_id , kp_id) : (" << i << ", " << id_2d << ")  ";
		}
	}
}

void bundleAdjustment() {
	
	vector<vector<KeyPoint>>& all_keypoints = trackInfo.all_keypoints;
	vector<Eigen::Isometry3d>& poses = trackInfo.poses;
	vector<Eigen::Isometry3d>& curPoses = trackInfo.curPoses;
	PointCloudT::Ptr pointCloud = trackInfo.pointCloud;
	map<int, vector<pair<int, int>>>& p3d_kf_kp = trackInfo.point3d_keyframe_keypoint;
	
	// 0 & preRealIndices & curRealIndices , all 7 keyframes' poses fixed
	unordered_set<int> fixedKF;
	unordered_set<int> isCurPoses;
	fixedKF.insert(0);
	for(int& id : trackInfo.preRealIndices) 
		fixedKF.insert(id);
	for(int& id : trackInfo.curRealIndices)
		fixedKF.insert(id), isCurPoses.insert(id);
	
	
	//get point3d map to <keyframe, keypoint>
	getPoint3d2KeyframeAndKeypoint();
	
	
	cout << "Output Bundle Adjustment Infomation ..." << endl << endl;
	cout << "trackInfo.all_keypoints.size() : " << all_keypoints.size() << endl;
	cout << "trackInfo.poses.size() : " << poses.size() << endl;
	cout << "trackInfo.curPoses.size() : " << curPoses.size() << endl;
	cout << "trackInfo.pointCloud->points.size() : " << pointCloud->points.size() << endl;
	cout << "trackInfo.point3d_keyframe_keypoint.size() : " << p3d_kf_kp.size() << endl;
	cout << endl << "Output Done." << endl << endl;
	
	// 步骤1：初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
	
	// 添加相机内参
	cout << "Adding camera matrix ..." << endl;
	Mat& K = trackInfo.K;
	g2o::CameraParameters* camera = new g2o::CameraParameters(
		fx, Eigen::Vector2d(cx, cy), 0 );
	camera->setId(0);
	optimizer.addParameter(camera);
	cout << "Done." << endl << endl;
	
	
	// 步骤2：向优化器添加顶点
	//vector<g2o::VertexSE3Expmap*> recSE3;

    // Set KeyFrame vertices
    // 步骤2.1：向优化器添加关键帧位姿顶点
	cout << "Adding pose vertex ..." << endl;
	int curId = 0;
	for(int i = 0; i < poses.size(); i++) {
		
		g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
		if(isCurPoses.count(i)) {
			Eigen::Isometry3d curPose = curPoses[curId++];
			vSE3->setEstimate(Eigen2SE3Quat(curPose));
		} else {
			vSE3->setEstimate(Eigen2SE3Quat(poses[i]));
		}
		vSE3->setId(i); //pose_id 和　poses下标　一一对应
		vSE3->setFixed(fixedKF.count(i));
		optimizer.addVertex(vSE3);
		//recSE3.push_back(vSE3);
	}
	cout << "Done." << endl << endl;
	
	const float thHuber2D = sqrt(5.99);
	const float thHuber3D = sqrt(7.815);
	const int poseSize = poses.size();
	
	// Set MapPoint vertices
    // 步骤2.2：向优化器添加MapPoints顶点
	cout << "Adding 3d point vertex and edges ..." << endl;
	vector<g2o::EdgeProjectXYZ2UV*> edges;
	int nEdges = 0;
	for(int i = 0; i < pointCloud->points.size(); i++) {
		
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate( PointT2Vector3d(pointCloud->points[i]) );
		int pt_id = poseSize + i;
		vPoint->setId(pt_id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);
		
		//SET EDGES
        // 步骤3：向优化器添加投影边边
		vector<pair<int, int>>& kf_kp = p3d_kf_kp[i];
		for(auto& kk : kf_kp) {
			int kf_id = kk.first;
			int kp_id = kk.second;
			
			KeyPoint& kp = all_keypoints[kf_id][kp_id];
			Eigen::Vector2d obs(kp.pt.x, kp.pt.y);
			
			g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
			edge->setId(nEdges++);
			edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pt_id)));
			edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(kf_id)));
			edge->setMeasurement(obs);
			edge->setParameterId(0, 0);
			edge->setInformation(Eigen::Matrix2d::Identity());
			optimizer.addEdge(edge);
			
			//huber kernel
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			edge->setRobustKernel(rk);
			rk->setDelta(thHuber2D);
			
			edges.push_back(edge);
		}
		
	}
	cout << "Done." << endl << endl;

#ifdef VERBOSE1	
	int in = 0;
	for(auto e : edges) {
		e->computeError();
		
		// chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
		if(e->chi2() > 1) {
			//cout << "error = " << e->chi2() << endl;
		} else {
			in++;
		}
	}
	cout<<"inliers size :  "<< in << endl <<endl;
#endif
	
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	
	optimizer.setVerbose(true);
	optimizer.initializeOptimization();
	optimizer.optimize(10);
	
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
	
#ifdef VERBOSE
	cout << "optimization costs time : " << time_used.count() << " seconds." << endl;
#endif
	
	cout << "get optimization result ..." << endl;
	// 步骤4:得到优化的结果
	vector<Eigen::Isometry3d> new_poses;
	for(int i = 0; i < poses.size(); i++) {
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		
		new_poses.push_back(toIsometry(SE3quat));
	}
	
	PointCloudT::Ptr new_pointCloud(new PointCloudT);
	for(int i = 0; i < pointCloud->points.size(); i++) {
		int id = poses.size() + i;
		
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(id));
		new_pointCloud->points.push_back(toPointT(vPoint->estimate()));
		
		//add color
		new_pointCloud->points.back().b = pointCloud->points[i].b;
		new_pointCloud->points.back().g = pointCloud->points[i].g;
		new_pointCloud->points.back().r = pointCloud->points[i].r;
	}
	
	trackInfo.new_poses = new_poses;
	trackInfo.new_pointCloud = new_pointCloud;
	cout << "Done." << endl << endl;
	
#ifdef VIZ
	int inliers = 0;
	for(auto e : edges) {
		e->computeError();
		
		// chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
		if(e->chi2() > 1) {
			//cout << "error = " << e->chi2() << endl;
		} else {
			inliers++;
		}
	}
	cout<<"inliers in total points: "<<inliers<<" / "<< pointCloud->points.size() <<endl;
	cout << endl << "all_points.size() : " << trackInfo.pointCloud->points.size() << endl;
	cout << "edges.size()      : " << edges.size() << endl << endl;
	
	// for visualization
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer_after"));
	viewer->addPointCloud(new_pointCloud, "Map_after");
	viewer->spin();
	
	while(!viewer->wasStopped()) {
		viewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
#endif
	
	
#ifdef LOAD
	cout << "save map point after optimization ..." << endl;
	pcl::io::savePCDFileBinary("after.pcd", *new_pointCloud);
	cout << "Done." << endl << endl;
#endif
	
}



int main(int argc, char** argv) {
	
	// target : features point cloud
	PointCloudT::Ptr tmpPointCloud(new PointCloudT);
 	trackInfo.pointCloud = tmpPointCloud; 
	
	string colorPath = "/home/tsinggoo/gu_dataTUM3/color/";
	string depthPath = "/home/tsinggoo/gu_dataTUM3/depth/";
	
	unsigned int N = 258; //TUM3
	if(argc == 2) {
		N = atoi( argv[1] );
	} else if(argc == 4) {
		N = atoi( argv[1] );
		colorPath = argv[2];
		depthPath = argv[3];
	}
	
	
	//KeyFrame Poses
	ifstream fin("/home/tsinggoo/gu_dataTUM3/KeyFramePos.txt");
	if(!fin) {
		cerr << "Wrong folder!" << endl;
		return 1;
	}
	
	//!!! include all tracking-needed information
	//TrackInfo trackInfo;
	
	// Loop record
	vector<int> preLoopIndices;
	vector<int> curLoopIndices;
	preLoopIndices.push_back(5), preLoopIndices.push_back(6), preLoopIndices.push_back(7);
	curLoopIndices.push_back(240), curLoopIndices.push_back(241), curLoopIndices.push_back(242);
	
	vector<int>& preRealIndices   = trackInfo.preRealIndices;
	vector<int>& curRealIndices   = trackInfo.curRealIndices;
	vector<Mat>& preLoopColorImgs = trackInfo.preLoopColorImgs;
	vector<Mat>& preLoopDepthImgs = trackInfo.preLoopDepthImgs;
	vector<Mat>& curLoopColorImgs = trackInfo.curLoopColorImgs;
	vector<Mat>& curLoopDepthImgs = trackInfo.curLoopDepthImgs;
	
	//hash map
	unordered_set<int> preLoopMark;
	unordered_set<int> curLoopMark;
	preLoopMark.insert(preLoopIndices[0]);
	preLoopMark.insert(preLoopIndices[1]);
	preLoopMark.insert(preLoopIndices[2]);
	curLoopMark.insert(curLoopIndices[0]);
	curLoopMark.insert(curLoopIndices[1]);
	curLoopMark.insert(curLoopIndices[2]);
	
	
	vector<Mat>& colorImgs                  = trackInfo.colorImgs;
	vector<Mat>& depthImgs                  = trackInfo.depthImgs;
	vector<vector<KeyPoint>>& all_keypoints = trackInfo.all_keypoints;
	vector<Mat>& descriptors                = trackInfo.descriptors;
	vector<Eigen::Isometry3d>& poses        = trackInfo.poses;
	//vector<DBoW3::BowVector> bowVecs;
	//vector<DBoW3::FeatureVector> featVecs;
	
	Ptr<Feature2D> detector = ORB::create(5000); //5000 features
	
	vector<unsigned int> validId;
	long unsigned int KFId;
	cout << "Loading color and depth images ..." << endl << endl;
	fin >> KFId;
	for(int i = 0; i <= N; i++) { //将Pos和图像对应，拿到图像的Mat矩阵，放入colorImgs depthImgs，拿到位姿放入poses
		
#ifdef VERBOSE
		//cout << "KFId: " << KFId << endl;
#endif
		if(i != KFId) continue;
		validId.push_back(KFId);
		
		boost::format fmt("/home/tsinggoo/gu_dataTUM3/%s/%d.%s");
		colorImgs.push_back(cv::imread( (fmt%"color"%(i)%"png").str() ));
		depthImgs.push_back(cv::imread( (fmt%"depth"%(i)%"png").str(), -1 ));
		
		double data[7] = {0};
		for(auto &d : data) fin >> d;
		
#ifdef VERBOSE
		//cout << "pose: " <<data[0]<<" "<<data[1]<<" "<<data[2]<<" "<<data[3]<<" "<<data[4]<<" "<<data[5]<<" "<<data[6]<< endl;
#endif	
		
		Eigen::Quaterniond q(data[6], data[3], data[4], data[5]); //姿态四元数w,ux,uy,uz
		Eigen::Isometry3d T(q);
		T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2])); //三个位置参数x,y,z
		poses.push_back(T);
		
		//compute keyponts & Descriptor
		vector<KeyPoint> keypoints;
		Mat descriptor;
		detector->detectAndCompute(colorImgs.back(), Mat(), keypoints, descriptor);
		all_keypoints.push_back(keypoints);
		descriptors.push_back(descriptor);
		//vector<Mat> vDescriptor;
		//getVecDesc(descriptor, vDescriptor);
		
		
		//record pre loop & cur loop
		if(preLoopMark.count(KFId)) {
			preLoopColorImgs.push_back(colorImgs.back());
			preLoopDepthImgs.push_back(depthImgs.back());
			
			preRealIndices.push_back(validId.size()-1);
		}
		
		if(curLoopMark.count(KFId)) {
			curLoopColorImgs.push_back(colorImgs.back());
			curLoopDepthImgs.push_back(depthImgs.back());
			
			curRealIndices.push_back(validId.size()-1);
		}
		
		if(i != N) fin >> KFId;

	} // load image done.
	cout << endl << "Done ..." << endl << endl;
	
	trackInfo.N = validId.size();
	Mat K = ( Mat_<double> ( 3,3 ) << fx, 0, cx, 0, fy, cy, 0, 0, 1 );
	trackInfo.K = K;
	
	cout << "trackByBoW() ..." << endl;
	trackByBoW();
	cout << "Done." << endl << endl;
	
#ifdef LOAD
	cout << "save map point before optimization ..." << endl;
	pcl::io::savePCDFileBinary("before.pcd", *trackInfo.pointCloud);
	cout << "Done." << endl << endl;
#endif
	
	cout << "loopPoseEstimationPnP() ..." << endl;
	loopPoseEstimationPnP();
	cout << "Done." << endl << endl;

#ifdef VIZ1	
	cout << "building map before bundle adjustment ..." << endl;
	buildMap();
	cout << "Done." << endl << endl;
#endif
	
	cout << "bundleAdjustment() ..." << endl;
	bundleAdjustment();
	cout << "Done." << endl << endl;
	
	
	
	
	/*************************** OUTUPT TEST ******************************/
#ifdef OUTPUT
	cout << "poses.size() vs new_poses.size() : " << poses.size() << " vs " << trackInfo.new_poses.size() << endl << endl;
	cout << "poses comparing ..." << endl << endl;
	for(int i = 0; i < poses.size(); i++) {
		cout << "***** pose " << i << " ***** " << endl;
		cout << "old :" << endl << poses[i].matrix() << endl;
		cout << "new :" << endl << trackInfo.new_poses[i].matrix() << endl;
	}
	
	cout << "pre loop before vs after ..." << endl;
	for(auto id : preRealIndices) {
		cout << "@@@@@ pose " << id << " @@@@@" << endl;
		cout << "old :" << endl << poses[id].matrix() << endl;
		cout << "new :" << endl << trackInfo.new_poses[id].matrix() << endl;
	}
	
	cout << endl << "cur loop before vs after ..." << endl;
	for(auto id : curRealIndices) {
		cout << "@@@@@ pose " << id << " @@@@@" << endl;
		cout << "old :" << endl << poses[id].matrix() << endl;
		cout << "new :" << endl << trackInfo.new_poses[id].matrix() << endl;
	}
	
	cout << "pointCloud.size() vs  new_pointCloud.size() : " << trackInfo.pointCloud->points.size() 
		 << " vs " << trackInfo.new_pointCloud->points.size() << endl;
	cout << endl << "compare selected some map points ..." << endl;
	for(int i = 0; i < trackInfo.pointCloud->points.size(); i++) {
		if(i % 400) continue;
		
		PointT& pt1 = trackInfo.pointCloud->points[i];
		PointT& pt2 = trackInfo.new_pointCloud->points[i];
		cout << "point id : " << i ;
		cout << "  old vs new : (" << pt1.x << ", " << pt1.y << ", " << pt1.z << ") vs (" << pt2.x << ", " << pt2.y << ", " << pt2.z << ")" << endl;
	}
	
	cout << endl << "output all poses ..." << endl << endl;
	for(int i = 0; i < trackInfo.poses.size(); i++) {
		if(i % 10) continue;
		
		Eigen::Isometry3d& T1 = trackInfo.poses[i];
		Eigen::Isometry3d& T2 = trackInfo.new_poses[i];
		cout << "pose id : " << i << endl;
		cout << "old pose: " << endl << T1.matrix() << endl;
		cout << "new pose: " << endl << T2.matrix() << endl << endl;
	}
	cout << "Done." << endl << endl;
#endif
	/**************************** BUILD MAP ******************************/
	
	vector<Eigen::Isometry3d>& new_poses = trackInfo.new_poses;
	PointCloudT::Ptr new_pointCloud(new PointCloudT);
	
	for(int i = 0; i < trackInfo.N; i++) {
		
		Mat& color = colorImgs[i];
		Mat& depth = depthImgs[i];
		Eigen::Isometry3d& T = new_poses[i];
		
		for(int v = 0; v < color.rows; v++) {
			for(int u = 0; u < color.cols; u++) {
				
				unsigned int d = depth.ptr<unsigned short>(v)[u];
				if(d == 0) continue;
				if(v%3 || u%3) continue;
				
				Eigen::Vector3d point;
				point[2] = double(d) / depthScale;
				if(point[2] > 3.0) continue;
				
				point[0] = (u-cx) * point[2] / fx;
				point[1] = (v-cy) * point[2] / fy;
				Eigen::Vector3d pointWorld = T * point;
				
				PointT p;
				p.x = pointWorld[0];
				p.y = pointWorld[1];
				p.z = pointWorld[2];
				p.b = color.data[v*color.step + u*color.channels()]; //这里注意颜色的顺序！！！
				p.g = color.data[v*color.step + u*color.channels() + 1];
				p.r = color.data[v*color.step + u*color.channels() + 2];
				
				new_pointCloud->points.push_back(p);
			}
		}
		
	}
	
	new_pointCloud->is_dense = false;
	
#ifdef VIZ2
	// for visualization
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer2"));
	viewer->addPointCloud(new_pointCloud, "new_Map");
	viewer->spin();
	
	while(!viewer->wasStopped()) {
		viewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
#endif
	
#ifdef LOAD
	cout << endl << "点云共有　" << new_pointCloud->size() << "　个点." << endl;
	cout << "save map points ..." << endl;
	pcl::io::savePCDFileBinary("map.pcd", *new_pointCloud);
	cout << "Done." << endl << endl;
#endif 
	
	return 0;
}