
//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>

//OpenCV
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointCloud<PointType> CloudType;

typedef std::vector<std::vector<double> > vec2d;


void loadCloudFromPNGImages(const std::string &inputdir, const std::string &filename, CloudType::Ptr cloud, cv::Mat &rgb, int pyramid_levels);
cv::Mat pyramid_downsample(cv::Mat image, int levels);
cv::Mat nanConv(cv::Mat img);

void extract_nonholesmask_from_pointcloud(const CloudType::Ptr cloud, cv::Mat &mask);
void extract_image_from_pointcloud(const CloudType::Ptr cloud, cv::Mat &rgb, std::string img_path, std::string dataset_type);
