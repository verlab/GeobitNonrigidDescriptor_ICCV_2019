#include "utils.hpp"
#include "clouds.hpp"

// libgeodesic
#include "hmTriDistance.h"
#include "hmContext.h"
#include "hmUtility.h"
#include "hmVectorSizeT.h"

void extract_descriptors_rotated(
    std::string cloud_path,
    const CloudType::Ptr cloud,
    std::vector<cv::KeyPoint> &keypoints,
    std::vector<cv::Mat> &descriptors,
    std::string sourcedir,
    std::string dataset_type,
    bool recalc_heatflow,
    int patchSize,
    int nb_angle_bins,
    std::string use_kp_orientation,
    float max_isocurve_size);

void extract_descriptors(std::string cloud_path,
                         const CloudType::Ptr cloud,
                         std::vector<cv::KeyPoint> &keypoints,
                         cv::Mat &descriptors,
                         std::string sourcedir,
                         std::string dataset_type,
                         bool recalc_heatflow,
                         int patchSize,
                         std::string use_kp_orientation,
                         float max_isocurve_size);

void compute_features_rotated(const CloudType::Ptr cloud, std::string img_path, const std::vector<std::vector<double>> &distance,
                              std::vector<cv::KeyPoint> &keypoints,
                              std::vector<cv::Mat> &rotated_descriptors,
                              std::string sourcedir,
                              std::string dataset_type,
                              int patchSize,
                              int nb_angle_bins,
                              std::string use_kp_orientation,
                              float max_isocurve_size);

void compute_heat_flow_c(const pcl::PolygonMesh &pcl_mesh,
                         std::vector<cv::KeyPoint> &coords_keypoints,
                         std::vector<std::vector<double>> &dist_heat_flow);

void filter_keypoints_on_hole(std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Mat> &img_distances);

void load_test_pairs(const std::string &filename, std::vector<std::vector<float>> &test_pairs);

float compute_canonical_orientation(const cv::Mat &image, cv::KeyPoint &keypoint,
                                    const std::vector<int> &umax, int patchSize, std::string use_kp_orientation);
void estimatePointAtIsoCurve(const cv::Mat &heat_flow, const cv::KeyPoint &kp, float dir, float isocurve, cv::Point2f &point, float max_isocurve_size);
int smoothedSum(const cv::Mat &sum, const cv::Point2f &pt, int KERNEL_SIZE = 9);
void remove_keypoints_on_holes(const pcl::PolygonMesh &pcl_mesh, std::vector<cv::KeyPoint> &keypoints, float scale);

void pclMesh2libgeodesicMesh(const pcl::PolygonMesh &pcl_mesh,
                             hmTriDistance *distance,
                             hmTriMesh *mesh,
                             std::vector<int> &inverse_shift_connected_vertices,
                             std::vector<int> &shifts);

bool setSources(hmTriDistance *distance, size_t keypoint_index);
std::vector<double> interpolate_heatflow(std::vector<double> &heatflow, float scale, cv::Point2f p);
cv::Point2f rotate2d(const cv::Point2f &inPoint, const float &angRad);
void apply_offset2d(cv::Mat &in, cv::Mat &out, int offsetx, int offsety);

void compute_vector_feature(const CloudType::Ptr cloud, std::string img_path, const std::vector<std::vector<double>> &distance,
                            std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat &descriptors,
                            std::string sourcedir,
                            std::string dataset_type,
                            int patchSize,
                            float max_isocurve_size,
                            std::string use_kp_orientation);

std::vector<cv::DMatch> calcAndSaveHammingDistancesNonrigid(std::vector<cv::KeyPoint> kp_query,
                                                            std::vector<cv::KeyPoint> kp_tgt,
                                                            std::vector<cv::Mat> desc_query,
                                                            std::vector<cv::Mat> desc_tgt,
                                                            CSVTable query,
                                                            CSVTable tgt,
                                                            std::string file_name,
                                                            int nb_angle_bins);

int norm_hamming_nonrigid(std::vector<cv::Mat> &src, std::vector<cv::Mat> &tgt, int idx_d1, int idx_d2, int nb_angle_bins);

void filter_matches(const std::vector<cv::DMatch> &matches, int threshold, std::vector<cv::DMatch> &filtered_matches);

std::vector<cv::DMatch> validate_matches(std::vector<cv::KeyPoint> &src, std::vector<cv::KeyPoint> &dst, std::vector<cv::DMatch> &matches, cv::Mat &img, int src_img_width);

std::vector<cv::DMatch> match_and_filter(std::vector<cv::Mat> d1, std::vector<cv::Mat> d2, std::vector<cv::KeyPoint> k1, std::vector<cv::KeyPoint> k2, std::string out_filename, int nb_angle_bins);

std::vector<cv::DMatch> calcAndSaveDistances(std::vector<cv::KeyPoint> kp_query,
                                             std::vector<cv::KeyPoint> kp_tgt,
                                             cv::Mat desc_query,
                                             cv::Mat desc_tgt,
                                             CSVTable query,
                                             CSVTable tgt,
                                             std::string file_name,
                                             int normType);
