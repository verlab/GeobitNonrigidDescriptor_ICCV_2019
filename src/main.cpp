#include <iostream>
#include <string>
#include <sstream>
#include <numeric>   // std::accumulate
#include <algorithm> // std::min_element, std::max_element
#include <map>
#include <chrono>
#include <stdio.h>

#include <math.h> // for M_PI

// PCL
#include <pcl/console/parse.h>

// libgeodesic
#include "hmTriDistance.h"
#include "hmContext.h"
#include "hmUtility.h"
#include "hmVectorSizeT.h"

#include "clouds.hpp"
#include "utils.hpp"
#include "geobit.hpp"

//#define DEBUG_ORIENTATION
#define USE_ROTATED_PATTERNS
#define SHOW_TEST_PAIRS 0
#define USE_KEYPOINT_ORIENTATION "ORB" // use "" to disable, "ORB" or "SURF" for ORB or SURF orientation estimation
#define RECALC_HEATFLOW 1

std::string dirnameOf(const std::string &fname)
{
    size_t pos = fname.find_last_of("\\/");
    return (std::string::npos == pos)
               ? ""
               : fname.substr(0, pos);
}

void print_help()
{
    std::cout << "" << std::endl;
    std::cout << "Usage: nonrigid_descriptor -inputdir INPUTDIR -refcloud REFCLOUD" << std::endl;
    std::cout << "                           -clouds CLOUD ... CLOUD -datasettype {real, simulated} " << std::endl;
    std::cout << "                           [-isocurvesize ISO] [-kpscale KPS] [-pyramidlevels PLV]" << std::endl;
    std::cout << "                           [-detector DET] [-distthreshold THR] " << std::endl
              << std::endl;
    std::cout << "-inputdir INPUTDIR        -> Directory containing depth and rgb images " << std::endl;
    std::cout << "-refcloud REFCLOUD        -> Name of reference cloud without extension " << std::endl;
    std::cout << "-clouds CLOUD .. CLOUD    -> List of clouds to compare with the reference cloud " << std::endl;
    std::cout << "-datasettype DTYPE        -> Dataset Type [real, simulated realdata-smoothed synthetic-smoothed] " << std::endl;
    std::cout << "-isocurvesize  ISO        -> Isocurve Size " << std::endl;
    std::cout << "-kpscale KPS              -> Keypoints Scale " << std::endl;
    std::cout << "-pyramidlevels PLV        -> Levels of Pyramid " << std::endl;
    std::cout << "-detector DET             -> Detector name " << std::endl;
    std::cout << "-distthreshold THR        -> Threshold Distance " << std::endl;
    std::cout << "" << std::endl;
}

bool cmdOptionExists(const char **begin, const char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char const *argv[])
{   
    //std::vector<cv::Mat> test;
    //save_hdf5_descs(test, "test.h5"); exit(0);

    if (cmdOptionExists(argv, argv + argc, "-h"))
    {
        print_help();
        return 0;
    }

    std::string inputdir;
    if (pcl::console::parse_argument(argc, argv, "-inputdir", inputdir) == -1)
    {
        PCL_ERROR("Need an input dir! Please use -inputdir to continue.\n");
        print_help();
        return (-1);
    }

    std::string refcloud;
    if (pcl::console::parse_argument(argc, argv, "-refcloud", refcloud) == -1)
    {
        PCL_ERROR("Need the refcloud file name! Please use -refcloud to continue.\n");
        print_help();
        return (-1);
    }

    std::vector<std::string> clouds;
    pcl::console::parse_multiple_arguments(argc, argv, "-clouds", clouds);

    std::string sourcedir;
    if (pcl::console::parse_argument(argc, argv, "-sourcedir", sourcedir) == -1)
    {
        sourcedir = dirnameOf(__FILE__);
    }

    float max_isocurve_size; // thickness of each isocurve

    float keypoint_scale = 7.0; //default OpenCV KeyPoint.size for the FAST detector
    pcl::console::parse_argument(argc, argv, "-kpscale", keypoint_scale);
    std::cout << "keypoint scale: " << keypoint_scale << std::endl;

    int pyramid_levels = 0;
    pcl::console::parse_argument(argc, argv, "-pyramidlevels", pyramid_levels);
    std::cout << "Pyramid Levels: " << pyramid_levels << std::endl;

    std::vector<std::string> descriptor_alg;
    pcl::console::parse_multiple_arguments(argc, argv, "-desc", descriptor_alg);

    std::string keypoint_detector;
    if (pcl::console::parse_argument(argc, argv, "-detector", keypoint_detector) == -1)
    {
        PCL_ERROR("Not defined a detector. Using Star detector.\n");
        keypoint_detector = "STAR";
    }

    double dist_threshold;
    if (pcl::console::parse_argument(argc, argv, "-distthreshold", dist_threshold) == -1)
    {
        PCL_ERROR("Not defined distance threshold. Using 15.\n");
        dist_threshold = 15;
    }

    std::string dataset_type;
    if (pcl::console::parse_argument(argc, argv, "-datasettype", dataset_type) == -1)
    {
        PCL_ERROR("Did not specified a dataset type. (real or simulated / smoothed or not).\n");
        //keypoint_detector = "STAR";
        print_help();
        return -1;
    }
    else //Set default support region size according to dataset type
    {
        if(dataset_type.find("real") != std::string::npos) // Metric scale data (RGB-D sensors)
            max_isocurve_size = 0.002;
        else
            max_isocurve_size = 0.06; //Simulated sequences
    }

    pcl::console::parse_argument(argc, argv, "-isocurvesize", max_isocurve_size);
    std::cout << "isocurve size: " << max_isocurve_size << std::endl;

    if (!strcmp(USE_KEYPOINT_ORIENTATION, ""))
        PCL_WARN("[WARNING]: Keypoint orientation normalization disabled for all descriptors.\n");

    // Geobit Config
    bool recalc_heatflow = false;
    int patchSize = 31;
    int nb_angle_bins = 12;
    std::string use_kp_orientation = "ORB";

    // Load RGB point cloud
    std::stringstream filename_input;
    filename_input << inputdir << "/" << refcloud;

    CloudType::Ptr ref_cloud(new CloudType); ref_cloud->header.seq = pyramid_levels;
    cv::Mat ref_img;
    cv::Mat ref_img_scaled;

    loadCloudFromPNGImages(inputdir, refcloud, ref_cloud, ref_img, pyramid_levels);

    // Load keypoint coordinates
    std::vector<cv::KeyPoint> ref_keypoints;
    CSVTable ref_groundtruth;

// Load Detector
#ifdef CV4 // use opencv4
    cv::Ptr<cv::FeatureDetector> feature_detector = cv::SIFT::create();
#else
    cv::Ptr<cv::FeatureDetector> feature_detector = cv::xfeatures2d::SIFT::create();
#endif

    //extract_image_from_pointcloud(ref_cloud, ref_img, filename_input.str() + "color.png");
    cv::resize(ref_img, ref_img_scaled, cv::Size(), keypoint_scale, keypoint_scale);

    cv::Mat nonholesmask;

    std::stringstream keypoints_filename;
    keypoints_filename << inputdir << "/" << refcloud << ".csv";

    bool use_detector = false;
    if (!load_groundtruth_keypoints_csv(keypoints_filename.str(), ref_keypoints, ref_groundtruth))
    {
        use_detector = true;
        extract_nonholesmask_from_pointcloud(ref_cloud, nonholesmask);
        feature_detector->detect(ref_img, ref_keypoints);

        //cv::KeyPointsFilter::runByImageBorder(ref_keypoints, cv::Size(640,480), 50);
        get_best_keypoints(ref_keypoints);
        std::cout << "#Detected keypoints [refcloud]: " << ref_keypoints.size() << std::endl
                  << std::flush;
        save_kps(ref_keypoints, refcloud + ".kp");
    }
    else
    {
        std::cout << "#Loaded keypoints [refcloud]: " << ref_keypoints.size() << std::endl
                  << std::flush;
    }

#ifdef USE_ROTATED_PATTERNS
    std::vector<cv::Mat> ref_descriptors;
    extract_descriptors_rotated(filename_input.str(), ref_cloud, ref_keypoints, ref_descriptors, sourcedir, dataset_type, recalc_heatflow,
                                patchSize, nb_angle_bins, use_kp_orientation, max_isocurve_size);
    //Save Descriptors
    save_hdf5_descs(ref_descriptors, ref_keypoints, filename_input.str() + ".h5");

#else
    cv::Mat ref_descriptors;
    extract_descriptors(filename_input.str(), ref_cloud, ref_keypoints, ref_descriptors, sourcedir, dataset_type, recalc_heatflow,
                        patchSize, use_kp_orientation, max_isocurve_size);

#endif



    for (size_t c = 0; c < clouds.size(); ++c)
    {

        // Load Cloud
        std::stringstream filename_cloud;
        filename_cloud << inputdir << "/" << clouds[c];

        CloudType::Ptr cloud(new CloudType); cloud->header.seq = pyramid_levels;
        cv::Mat img2;
        cv::Mat img2_scaled;

        loadCloudFromPNGImages(inputdir, clouds[c], cloud, img2, pyramid_levels);
        cv::resize(img2, img2_scaled, cv::Size(), keypoint_scale, keypoint_scale);

        // Load keypoint coordinates
        std::stringstream keypoints_filename;
        keypoints_filename << inputdir << "/" << clouds[c] << ".csv";

        std::vector<cv::KeyPoint> keypoints;
        CSVTable groundtruth;

        if (!load_groundtruth_keypoints_csv(keypoints_filename.str(), keypoints, groundtruth))
        {
            cv::Mat nonholesmask;
            extract_nonholesmask_from_pointcloud(cloud, nonholesmask);

            feature_detector->detect(img2, keypoints);

            use_detector = true;
            get_best_keypoints(keypoints);
            std::cout << "#Detected keypoints [cloud]: " << keypoints.size() << std::endl
                      << std::flush;
            save_kps(keypoints, clouds[c] + ".kp");
        }
        else
        {
            std::cout << "#Loaded keypoints [dst_clouds]: " << keypoints.size() << std::endl
                      << std::flush;
        }

        std::stringstream out_distances;
#ifdef USE_ROTATED_PATTERNS
        std::vector<cv::Mat> descriptors;
        extract_descriptors_rotated(filename_cloud.str(), cloud, keypoints, descriptors, sourcedir, dataset_type, recalc_heatflow,
                                    patchSize, nb_angle_bins, use_kp_orientation, max_isocurve_size);
        save_hdf5_descs(descriptors, keypoints, filename_cloud.str() + ".h5");

        std::cout << "Matches: RefCloud vs " << clouds[c] << " with sizes " << ref_descriptors[0].rows << " " << descriptors[0].rows << std::endl
                  << std::flush;
        out_distances << refcloud << "__" << clouds[c] << "__OURS";

#else
        cv::Mat descriptors;
        extract_descriptors(filename_cloud.str(), cloud, keypoints, descriptors, sourcedir, dataset_type, recalc_heatflow,
                            patchSize, use_kp_orientation, max_isocurve_size);

        std::cout << "Matches: RefCloud vs " << clouds[c] << " with sizes " << ref_descriptors.rows << " " << descriptors.rows << std::endl
                  << std::flush;
        out_distances << refcloud << "__" << clouds[c] << "__OURS";
#endif

        //Matcher
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> o_matches, matches;
        int threshold = dist_threshold;

        cv::Mat outimg;

        if (!use_detector)
        {
#ifdef USE_ROTATED_PATTERNS

            o_matches = calcAndSaveHammingDistancesNonrigid(ref_keypoints, keypoints, ref_descriptors, descriptors,
                                                            ref_groundtruth, groundtruth, out_distances.str() + ".txt", nb_angle_bins);

#else

            o_matches = calcAndSaveDistances(ref_keypoints, keypoints, ref_descriptors, descriptors,
                                             ref_groundtruth, groundtruth, out_distances.str() + ".txt", cv::NORM_HAMMING);

#endif

            filter_matches(o_matches, threshold, matches);
            cv::drawMatches(ref_img, ref_keypoints, img2, keypoints, matches, outimg);

            std::vector<cv::DMatch> valid_matches = validate_matches(ref_keypoints, keypoints, matches, outimg, ref_img.cols);

            //cv::imshow("ours", outimg);
            std::stringstream out_text;

            out_text << "OURS. # Correct: " << valid_matches.size();
            out_text << " / Acc.: " << std::fixed << std::setprecision(2) << 100.0 * (valid_matches.size() / (float)matches.size()) << " %"
                     << " / Tot.: " << matches.size();

            cv::putText(outimg, out_text.str().c_str(), cv::Point2f(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL,
                        0.8, cv::Scalar(90, 250, 90), 1, CV_AA);

            cv::imwrite(out_distances.str() + ".png", outimg);
            //cv::waitKey();
        }
        else
        {
            matches = match_and_filter(ref_descriptors, descriptors, ref_keypoints, keypoints, out_distances.str() + ".txt", nb_angle_bins); // match and save .txt with correspondences
            cv::drawMatches(ref_img, ref_keypoints, img2, keypoints, matches, outimg);
            cv::imwrite(out_distances.str() + ".png", outimg);
        }

        std::cout << "------------- Now comparing with #" << descriptor_alg.size() << " baselines -------------" << std::endl;

        for (size_t d = 0; d < descriptor_alg.size(); ++d)
        {
            std::cout << "*** Baseline descriptor *** " << descriptor_alg[d] << std::endl;

            cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;

            std::vector<cv::KeyPoint> ref_kps, kps;

            ref_kps = ref_keypoints;
            kps = keypoints;

            for (int i = 0; i < ref_kps.size(); i++)
            {
                ref_kps[i].pt.x *= keypoint_scale;
                ref_kps[i].pt.y *= keypoint_scale;
            }

            for (int i = 0; i < kps.size(); i++)
            {
                kps[i].pt.x *= keypoint_scale;
                kps[i].pt.y *= keypoint_scale;
            }

            printf("Keypoint Scale = %.2f, #keypoints = %d\n\n", ref_kps[0].size, ref_kps.size());

            cv::Mat descriptors_ref, descriptors_img2;

            if (descriptor_alg[d] == "FREAK")
            {
                descriptor_extractor = cv::xfeatures2d::FREAK::create();
            }
            else if (descriptor_alg[d] == "BRIEF")
            {
                descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
            }
            else if (descriptor_alg[d] == "BRISK")
            {
                descriptor_extractor = cv::BRISK::create();
            }
            else if (descriptor_alg[d] == "ORB")
            {
                descriptor_extractor = cv::ORB::create();
            }
            else if (descriptor_alg[d] == "DAISY")
            {
                descriptor_extractor = cv::xfeatures2d::DAISY::create(15.0, 3, 8, 8, cv::xfeatures2d::DAISY::NRM_NONE, cv::noArray(), true, true);
            }else{
                std::cout << "Invalid descriptor: " << descriptor_alg[d] << std::endl;
                continue;
            }   

            std::vector<cv::DMatch> o_matches_competitor, matches_competitor;

            if (!use_detector)
            {
                descriptor_extractor->compute(ref_img_scaled, ref_kps, descriptors_ref);
                descriptor_extractor->compute(img2_scaled, kps, descriptors_img2);

                cv::BFMatcher matcher2;

                if (descriptor_alg[d] == "DAISY")
                    matcher2 = cv::BFMatcher(cv::NORM_L1, true);
                else
                    matcher2 = cv::BFMatcher(cv::NORM_HAMMING, true);

                std::stringstream out_distances;
                out_distances << refcloud << "__" << clouds[c] << "__" << descriptor_alg[d];

                if (descriptor_alg[d] == "DAISY")
                    o_matches_competitor = calcAndSaveDistances(ref_kps, kps, descriptors_ref, descriptors_img2,
                                                                ref_groundtruth, groundtruth, out_distances.str() + ".txt", cv::NORM_L1);
                else
                    o_matches_competitor = calcAndSaveDistances(ref_kps, kps, descriptors_ref, descriptors_img2,
                                                                ref_groundtruth, groundtruth, out_distances.str() + ".txt", cv::NORM_HAMMING);

                if (descriptor_alg[d] == "BRISK") //longer descriptor in size
                    filter_matches(o_matches_competitor, threshold * 2, matches_competitor);
                else
                    filter_matches(o_matches_competitor, threshold, matches_competitor);

                int correct = 0;
                std::cout << descriptor_alg[d] << " - Matches: RefCloud vs " << clouds[c] << std::endl
                          << std::flush;

                cv::drawMatches(ref_img_scaled, ref_kps, img2_scaled, kps, matches_competitor, outimg);

                std::vector<cv::DMatch> valid_matches = validate_matches(ref_kps, kps, matches_competitor, outimg, ref_img_scaled.cols);

                std::stringstream out_text;

                out_text << descriptor_alg[d] << ". # Correct: " << valid_matches.size();
                out_text << " / Acc.: " << std::fixed << std::setprecision(2) << 100.0 * (valid_matches.size() / (float)matches_competitor.size()) << " %"
                         << " / Tot.: " << matches_competitor.size();

                cv::putText(outimg, out_text.str().c_str(), cv::Point2f(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL,
                            0.8, cv::Scalar(90, 250, 90), 1, CV_AA);

                cv::imwrite(out_distances.str() + ".png", outimg);

                //std::cout<<descriptors_ref.row(50)<<std::endl;
                std::cout << "-------------------------------" << std::endl;

                //cv::waitKey();
            }
        }
    }

    return 0;
}
