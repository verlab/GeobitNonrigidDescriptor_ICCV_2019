#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <numeric>      // std::accumulate
#include <algorithm>    // std::min_element, std::max_element
#include <map>
#include <chrono>
#include <stdio.h>

//OpenCV
#include "opencv2/core.hpp"

//PCL
#include <pcl/console/print.h>

// Opencv 4 compatibility
#define CV4
#ifdef CV4

#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_INTER_LINEAR cv::INTER_LINEAR
#define CV_AA cv::LINE_AA

#endif

typedef std::vector<std::map<std::string,float> > CSVTable;
typedef std::vector<std::vector<double> > vec2d;

bool load_groundtruth_keypoints_csv(const std::string &filename, std::vector<cv::KeyPoint> &keypoints, CSVTable &csv_data);
void get_best_keypoints(std::vector<cv::KeyPoint>& kps);
void filter_kps_boundingbox(std::vector<cv::KeyPoint>& kps, cv::Point2f pmin, cv::Point2f pmax);
bool sort_kps(cv::KeyPoint k1, cv::KeyPoint k2);
void save_kps(std::vector<cv::KeyPoint> keypoints, std::string out_filename);
bool load_heatflow_from_file(std::string filename, vec2d &heatflow);
void dump_heatflow_to_file(std::string filename, vec2d heatflow);

