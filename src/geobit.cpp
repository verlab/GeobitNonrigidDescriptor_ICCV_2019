#include "geobit.hpp"

void extract_descriptors(std::string cloud_path,
                         const CloudType::Ptr cloud,
                         std::vector<cv::KeyPoint> &keypoints,
                         cv::Mat &descriptors,
                         std::string sourcedir,
                         std::string dataset_type,
                         bool recalc_heatflow,
                         int patchSize,
                         std::string use_kp_orientation,
                         float max_isocurve_size)
{
    printf("Extracting descriptors from %d keypoints.\n", keypoints.size());
    CloudType::Ptr filtered_cloud(new CloudType);
    //depth_filtering(cloud, filtered_cloud);

    // create a mesh from the an organized point cloud
    pcl::PolygonMesh mesh;

    pcl::OrganizedFastMesh<PointType> ofm;

    // Set parameters
    ofm.setInputCloud(cloud);
    if (dataset_type.find("real") != std::string::npos)
        ofm.setMaxEdgeLength(1.0);
    else
        ofm.setMaxEdgeLength(10.0);

    ofm.setTrianglePixelSize(1);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<PointType>::TRIANGLE_ADAPTIVE_CUT);

    // Reconstruct
    std::cout << "Creating mesh... " << std::flush;
    ofm.reconstruct(mesh);
    mesh.header.seq = cloud->header.seq; //Pyramid Level
    std::cout << "done\n"
              << std::flush;

    std::vector<std::vector<double>> dist_heat_flow;

    if (recalc_heatflow || !load_heatflow_from_file(cloud_path + ".heatflow", dist_heat_flow))
    {
        printf("Computing heatflow from scratch... it may take a while\n");
        dist_heat_flow.clear();
        compute_heat_flow_c(mesh, keypoints, dist_heat_flow);
        //dump_heatflow_to_file(cloud_path + ".heatflow", dist_heat_flow); //save heatflow to disk - uncomment to allow precomputation
    }
    else
        printf("Loaded a precomputed heatflow from file!\n");

    compute_vector_feature(cloud, cloud_path + "color.png", dist_heat_flow, keypoints, descriptors, sourcedir, dataset_type, patchSize, max_isocurve_size, use_kp_orientation);
}

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
    float max_isocurve_size)
{

    auto start = std::chrono::steady_clock::now();

    printf("Extracting descriptors from %d keypoints.\n", keypoints.size());
    CloudType::Ptr filtered_cloud(new CloudType);
    //depth_filtering(cloud, filtered_cloud);

    // create a mesh from the an organized point cloud
    pcl::PolygonMesh mesh;

    pcl::OrganizedFastMesh<PointType> ofm;

    // Set parameters
    ofm.setInputCloud(cloud);
    if (dataset_type.find("real") != std::string::npos)
        ofm.setMaxEdgeLength(1.0); // 1.0
        
    else
        ofm.setMaxEdgeLength(10.0);

    ofm.setTrianglePixelSize(1);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<PointType>::TRIANGLE_ADAPTIVE_CUT);

    // Reconstruct
    std::cout << "Creating mesh... " << std::flush;
    ofm.reconstruct(mesh);
    std::cout << "done\n"
              << std::flush;

    std::vector<std::vector<double>> dist_heat_flow;

    if (recalc_heatflow || !load_heatflow_from_file(cloud_path + ".heatflow", dist_heat_flow))
    {
        dist_heat_flow.clear();
        printf("Computing heatflow from scratch... it may take a while\n");
        compute_heat_flow_c(mesh, keypoints, dist_heat_flow);
        //dump_heatflow_to_file(cloud_path + ".heatflow", dist_heat_flow); //uncomment this if you want to save precomputed heatflow to disk
    }
    else
        printf("Loaded a precomputed heatflow from file!\n");

    compute_features_rotated(cloud, cloud_path + "-rgb.png", dist_heat_flow, keypoints, descriptors, sourcedir, dataset_type, patchSize, nb_angle_bins, use_kp_orientation, max_isocurve_size);
}

void compute_features_rotated(const CloudType::Ptr cloud, std::string img_path, const std::vector<std::vector<double>> &distance,
                              std::vector<cv::KeyPoint> &keypoints,
                              std::vector<cv::Mat> &rotated_descriptors,
                              std::string sourcedir,
                              std::string dataset_type,
                              int patchSize,
                              int nb_angle_bins,
                              std::string use_kp_orientation,
                              float max_isocurve_size)
{

    rotated_descriptors.clear();
    std::vector<cv::Mat> mat_distances_vec;
    int halfPatchSize = patchSize / 2;

    cv::Mat img = cv::imread(img_path.c_str());
    //extract_image_from_pointcloud(cloud, img, img_path, dataset_type);

    for (size_t kp = 0; kp < keypoints.size(); ++kp)
    {
        cv::Mat img_distance(img.rows, img.cols, CV_64FC1);

        //printf("rows cols %d %d size %d", img.rows, img.cols,distance[kp].size() ); getchar();

        for (int d = 0; d < distance[kp].size(); ++d)
        {
            double dist = distance[kp][d];

            int x = d % img.cols;
            int y = d / img.cols;

            img_distance.at<double>(y, x) = dist;
        }

        mat_distances_vec.push_back(img_distance);
    }

    filter_keypoints_on_hole(keypoints, mat_distances_vec);

    //Rotated versions of the descriptors (position 0 is the unrotated pattern

    //const std::string test_pairs_file = sourcedir + "aux/test_pairs_reaching_holes.txt"; // "test_pairs.txt";
    const std::string test_pairs_file = sourcedir + "/aux/gaussian_1024.txt";
    std::vector<std::vector<float>> test_pairs;

    load_test_pairs(test_pairs_file, test_pairs);

    int num_bytes = test_pairs.size() / 8;
    //descriptors = cv::Mat::zeros(keypoints.size(), num_bytes, CV_8U);

    cv::Mat grayImage = img;
    if (img.type() != CV_8U)
        cvtColor(img, grayImage, CV_BGR2GRAY);

    // Construct integral image for fast smoothing (box filter)
    cv::Mat sum;
    cv::integral(grayImage, sum, CV_32S);

    // pre-compute the end of a row in a circular patch (used for canonicall orientation)
    std::vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    for (int angle = 0; angle < nb_angle_bins; angle++)
    {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> valid_keypoints;

        printf("Calculating descriptors with rotation = %.2f (%d of %d)\n", angle * (360.f / (double)nb_angle_bins), angle + 1, nb_angle_bins);

        //cv::namedWindow("test_pairs", cv::WINDOW_NORMAL);
        for (size_t kp = 0; kp < keypoints.size(); ++kp)
        {

            cv::Mat img_distance = mat_distances_vec[kp];

            float canonical_angle = (angle * (360.f / (double)nb_angle_bins)) * (CV_PI / 180.f);
            compute_canonical_orientation(grayImage, keypoints[kp], umax, patchSize, use_kp_orientation);

#if SHOW_TEST_PAIRS
            // shows the keypoint orientation
            std::cout << "Canonical angle: " << canonical_angle * 180.0f / CV_PI << std::endl
                      << std::flush;
            std::cout << "KeyPoint: " << keypoints[kp].pt << std::endl
                      << std::flush;
            std::cout << "Dist: " << img_distance.at<double>(keypoints[kp].pt) << std::endl
                      << std::flush;

            cv::Point2f end_arrow = rotate2d(cv::Point2f(30, 0), canonical_angle);
            cv::arrowedLine(img, keypoints[kp].pt, keypoints[kp].pt + end_arrow, cv::Scalar(255, 0, 0), 1);
#endif

            // Keypoint on a hole (there is no geodesic info for it)
            cv::KeyPoint _keypoint = keypoints[kp];
            _keypoint.pt.x = (int)(keypoints[kp].pt.x + 0.5);
            _keypoint.pt.y = (int)(keypoints[kp].pt.y + 0.5);

            cv::Mat keypoint_descriptor = cv::Mat::zeros(1, num_bytes, CV_8U);
            uchar *desc = keypoint_descriptor.ptr(0);

            char bitstring[4096] = "";
            uchar feat = 0;

            for (size_t tp = 0; tp < test_pairs.size(); ++tp)
            {
                //std::cout << "=== Test pair # " << tp << "===" << std::endl << std::flush;

                float isocurve = test_pairs[tp][0];
                cv::Point2f pointX;
                float dir = canonical_angle + test_pairs[tp][1];
                estimatePointAtIsoCurve(img_distance, _keypoint, dir, isocurve, pointX, max_isocurve_size);

                isocurve = test_pairs[tp][2];
                cv::Point2f pointY;
                dir = canonical_angle + test_pairs[tp][3];
                estimatePointAtIsoCurve(img_distance, _keypoint, dir, isocurve, pointY, max_isocurve_size);

#if SHOW_TEST_PAIRS
                cv::line(img, pointX, pointY, cv::Scalar(0, 0, 255));
                cv::imshow("test_pairs", img);
                //cv::waitKey();
#endif
                uchar vtest = smoothedSum(sum, pointX) < smoothedSum(sum, pointY);
                //bool gtest = abs(img_distance.at<float>(pointX) - img_distance.at<float>(pointY)) > 0.001;

                uchar test = vtest; //| gtest;

                sprintf(bitstring, "%s %d", bitstring, test);
                feat = feat + (test << (tp % 8));
                //std::cout << std::endl << (int)feat;
                if ((tp + 1) % 8 == 0)
                {
                    desc[tp / 8] = feat;
                    feat = 0;
                    //std::cout << std::endl << bitstring << " (" << (int)desc[tp/8] << ")"<< std::endl;
                    //bitstring[0] = '\0';
                    //getchar();
                }
            }

            descriptors.push_back(keypoint_descriptor);
        }

        rotated_descriptors.push_back(descriptors);
    }
}

void compute_heat_flow_c(const pcl::PolygonMesh &pcl_mesh,
                         std::vector<cv::KeyPoint> &coords_keypoints,
                         std::vector<std::vector<double>> &dist_heat_flow)
{

    /* main data */
    hmContext context;
    hmTriMesh surface;
    hmTriDistance distance;
    distance.verbose = 0;

    /* initialize data */
    //std::cout << "hmContextInitialize..." << std::endl << std::flush;
    hmContextInitialize(&context);
    //std::cout << "hmTriMeshInitialize..." << std::endl << std::flush;
    hmTriMeshInitialize(&surface);
    // std::cout << "hmTriDistanceInitialize..." << std::endl << std::flush;
    hmTriDistanceInitialize(&distance);

    /* read surface */
    //std::cout << "pclMesh2libgeodesicMesh..." << std::endl << std::flush;
    std::vector<int> inverse_shift_connected_vertices;
    std::vector<int> shifts;
    float f_scale = 1.0 / pow(2, pcl_mesh.header.seq); // pcl_mesh.cloud.width / 640.0;

    remove_keypoints_on_holes(pcl_mesh, coords_keypoints, f_scale);

    pclMesh2libgeodesicMesh(pcl_mesh, &distance, &surface, inverse_shift_connected_vertices, shifts);

    /* set time for heat flow */
    // std::cout << "hmTriDistanceEstimateTime..." << std::endl << std::flush;
    hmTriDistanceEstimateTime(&distance);

    /*
   fprintf( stdout, "   -boundary [boundaryConditions]\n" );
   fprintf( stdout, "         Specify boundary conditions as a floating-point value between 0 and\n" );
   fprintf( stdout, "         1, inclusive.  A value of 0 yields pure-Neumann conditions; a value\n" );
   fprintf( stdout, "         of 1 yields pure Dirichlet conditions.  For typical usage (small\n" );
   fprintf( stdout, "         flow time or surfaces without boundary) this parameter has little\n" );
   fprintf( stdout, "         effect and optimal performance will be achieved by setting it to the\n" );
   fprintf( stdout, "         default value of zero.  For surfaces with boundary where a large flow\n" );
   fprintf( stdout, "         time is desired (i.e., smoothed geodesic distance) a value of 0.5\n" );
   fprintf( stdout, "         tends to give natural-looking behavior near the boundary.\n" );
    */

    double boundaryConditions = 0.5;
    hmTriDistanceSetBoundaryConditions(&distance, boundaryConditions);

    /* compute distance */
    std::cout << "hmTriDistanceBuild..." << std::endl
              << std::flush;
    hmTriDistanceBuild(&distance);
    std::cout << "distance.time: " << distance.time << std::endl
              << std::flush;

    for (size_t kp = 0; kp < coords_keypoints.size(); ++kp)
    {

        cv::KeyPoint _kp = coords_keypoints[kp];
        //_kp.pt*=f_scale;

        _kp.pt.x = (int)(coords_keypoints[kp].pt.x + 0.5);
        _kp.pt.y = (int)(coords_keypoints[kp].pt.y + 0.5);

        //printf("kp %d: (%f, %f)\n", kp, _kp.pt.x, _kp.pt.y);

        _kp.pt *= f_scale;

        _kp.pt.x = (int)(_kp.pt.x);
        _kp.pt.y = (int)(_kp.pt.y);

        size_t keypoint_index = _kp.pt.x + _kp.pt.y * pcl_mesh.cloud.width;

        size_t shifted_kpt_index = keypoint_index - shifts[keypoint_index];

        //std::cout << "setSources..." << std::endl << std::flush;
        setSources(&distance, shifted_kpt_index);

        /* udpate distance function */
        // std::cout << "hmTriDistanceUpdate..." << std::endl <<  std::flush;
        hmTriDistanceUpdate(&distance);

        /* write distances */
        // std::cout << "saving distances..." << std::endl <<  std::flush;
        std::vector<double> dists(shifts.size(), -1);

        for (size_t v = 0; v < inverse_shift_connected_vertices.size(); ++v)
        {
            dists[v + inverse_shift_connected_vertices[v]] = distance.distance.values[v];
        }
        // interpolate distance map

        std::vector<double> hd_dists;
        hd_dists = interpolate_heatflow(dists, f_scale, pcl_mesh.cloud.width, coords_keypoints[kp].pt);

        dist_heat_flow.push_back(hd_dists);
    }
}

void filter_keypoints_on_hole(std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Mat> &img_distances)
{
    // Keypoint on a hole (there is no geodesic info for it)

    std::vector<cv::KeyPoint> keypoints_;
    std::vector<cv::Mat> img_distances_;

    for (size_t kp = 0; kp < keypoints.size(); kp++)
    {
        cv::Mat img_distance = img_distances[kp];
        cv::KeyPoint _keypoint = keypoints[kp];
        _keypoint.pt.x = (int)(keypoints[kp].pt.x + 0.5);
        _keypoint.pt.y = (int)(keypoints[kp].pt.y + 0.5);

        if (std::isnan(img_distance.at<double>(_keypoint.pt)) || //std::fpclassify(img_distance.at<double>(_keypoint.pt)) != FP_ZERO ||
            img_distance.at<double>(_keypoint.pt) < 0)
        {
            printf("Keypoint with invalid distance: %f!\n", img_distance.at<double>(_keypoint.pt));
            std::cout << "Invalid distance: " << _keypoint.pt /*<< "\t\t\t 3D: " 
            << cloud->at(_keypoint.pt.x, _keypoint.pt.y) 
            << "\t\tDistance: " << img_distance.at<double>(_keypoint.pt) */
                      << std::endl
                      << std::flush;

            //show_keypoint_on_heatflow(cloud, distance[kp], keypoints[kp]);
            //cv::circle(img, _keypoint.pt, 3, cv::Scalar(0, 0, 0), 1);
            continue;
        }

        //show_keypoint_on_heatflow(cloud, img_distance, keypoints[kp]);

        keypoints_.push_back(_keypoint);
        img_distances_.push_back(img_distances[kp]);
    }

    keypoints = keypoints_;
    img_distances = img_distances_;
}

void load_test_pairs(const std::string &filename, std::vector<std::vector<float>> &test_pairs)
{
    std::ifstream fin(filename.c_str());
    if (fin.is_open())
    {
        while (!fin.eof())
        {
            std::vector<float> r_and_theta(4);

            fin >> r_and_theta[0];
            //std::cout << " r1 " << r_and_theta[0] << " ";
            fin >> r_and_theta[1]; // Point 1 (radius, theta)
            //std::cout << " theta1 " << r_and_theta[1] << " ";

            fin >> r_and_theta[2];
            //std::cout << " r2 " << r_and_theta[2] << " ";

            fin >> r_and_theta[3]; // Point 2 (radius, theta)
            //std::cout << " theta2 " << r_and_theta[3] << "\n";

            if (!fin.eof())
                test_pairs.push_back(r_and_theta);
        }
        fin.close();

        std::cout << "Loaded " << test_pairs.size() << " pair tests from the file." << std::endl;
    }
    else
    {
        PCL_ERROR("ERROR LOADING THE PAIRS TXT FILE...\n");
    }
}

// Compute the canonical orientation based on the Intensity Centroid approach
// Return the angle in radian
float compute_canonical_orientation(const cv::Mat &image, cv::KeyPoint &keypoint,
                                    const std::vector<int> &umax, int patchSize, std::string use_kp_orientation)
{
    // extracted from OpenCV (ORB.cpp and features2d.hpp)
    // static float IC_Angle(const Mat& image, const int half_k, Point2f pt, const vector<int> & u_max)
    int halfPatchSize = patchSize / 2;
    int m_01 = 0, m_10 = 0;

    const uchar *center = &image.at<uchar>(cvRound(keypoint.pt.y), cvRound(keypoint.pt.x));

    // Treat the center line differently, v=0
    for (int u = -halfPatchSize; u <= halfPatchSize; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= halfPatchSize; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = umax[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    float orientation = 0;
    keypoint.angle = -1;

    if (!strcmp(use_kp_orientation.c_str(), "ORB"))
    {
        orientation = (cv::fastAtan2((float)m_01, (float)m_10) * (float)(CV_PI / 180.f)); //orientation used by the non_rigid_desc
        keypoint.angle = cv::fastAtan2((float)m_01, (float)m_10);                         //orientation used by other descriptors
    }
    //
    //keypoint.angle = cv::fastAtan2((float)m_01, (float)m_10);
    //return orientation;
    #ifndef NO_CONTRIB
    else if (!strcmp(use_kp_orientation.c_str(), "SURF"))
    {
        //cv::Ptr<cv::Feature2D> surf = cv::SURF::create();//cv::Algorithm::create<cv::Feature2D>("Feature2D.SURF");
        //if (surf.empty())
        //    CV_Error(CV_StsNotImplemented, "OpenCV was built without SURF support.");

        cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(); // surf = cv::DescriptorExtractor::create("SURF");
        std::vector<cv::KeyPoint> keypoints;
        keypoints.push_back(keypoint);
        cv::Mat d;
        //std::cout << "**angle " << keypoint.pt << image.rows << " " << image.cols << std::endl;
        surf->compute(image, keypoints, d);
        //surf->operator()(image, cv::noArray(), keypoints, cv::noArray(), true);

        orientation = keypoints[0].angle * (float)(CV_PI / 180.f);
        keypoint.angle = keypoints[0].angle;

        // std::cout << "**angle " << keypoint.angle << std::endl;
    }
    #endif

    return orientation;
}

void estimatePointAtIsoCurve(const cv::Mat &heat_flow, const cv::KeyPoint &kp, float dir, float isocurve, cv::Point2f &point, float max_isocurve_size)
{
    cv::Point2f dir_vec = rotate2d(cv::Point2f(1, 0), dir);
    float step = 0.05;

    static int counter = 0;

    //std::cout << "dir: " << dir_vec << std::endl << std::flush;

    int id_isocurve = (int)(isocurve);

    int curr_isocurve = 0;

    double curr_distance = 0;
    double last_valid_dist = 0;

    cv::Vec2f k(kp.pt.x, kp.pt.y);

    // take a iso curve equal to the iso curve used in the sampling or the next one (when the
    // point is on a hole).
    point = kp.pt;

    cv::Point2f last_valid_pt = kp.pt;

    while (curr_isocurve < id_isocurve)
    {
        point = point + step * dir_vec;

        int img_y = (int)(point.y + 0.5);
        int img_x = (int)(point.x + 0.5);
        cv::Vec2f p(img_x, img_y);

        //std::cout << point << std::endl;
        if (cv::norm(p - k) > 24)
        {
            //printf("breaking the loop large norm: norm: %f\n", cv::norm(p-k));
            //getchar();
            //break;
        }

        if (point.x < 0 || img_x >= heat_flow.cols || point.y < 0 || img_y >= heat_flow.rows)
        {
            //PCL_ERROR("breaking the loop beyond the limits\n");
            //getchar();
            break;
        }

        if (!std::isnan(heat_flow.at<double>(point)) &&
            //std::fpclassify(heat_flow.at<double>(point)) != FP_ZERO)
            true) //heat_flow.at<double>(point) > 0.02)
        {
            curr_distance = heat_flow.at<double>(point);
            last_valid_dist = curr_distance; // Save the last valid point
            last_valid_pt.x = point.x;
            last_valid_pt.y = point.y;
        }

        curr_isocurve = (int)(curr_distance / max_isocurve_size);
    }

    if (!(point.x == last_valid_pt.x && point.y == last_valid_pt.y) && curr_distance - last_valid_dist != 0) // if they differ, we want to estimate a middle point in the hole
                                                                                                             //that will best explain the position in the RGB image
    {
        cv::Point2f middle_point = point - last_valid_pt;
        double ratio = ((id_isocurve * max_isocurve_size) - last_valid_dist) / (curr_distance - last_valid_dist);
        middle_point = last_valid_pt + (middle_point * ratio);

        point.x = middle_point.x;
        point.y = middle_point.y;

        if (point.x < 0 || point.x >= heat_flow.cols || point.y < 0 || point.y >= heat_flow.rows)
        {
            PCL_ERROR("breaking the loop beyond the limits\n");
            //getchar();
        }
    }
}

int smoothedSum(const cv::Mat &sum, const cv::Point2f &pt, int KERNEL_SIZE)
{
    // Values used by BRIEF
    //static const int PATCH_SIZE = 48;
    //static const int KERNEL_SIZE = 9;

    static const int HALF_KERNEL = KERNEL_SIZE / 2;

    //int img_y = (int)(kpt.pt.y + 0.5) + pt.y;
    //int img_x = (int)(kpt.pt.x + 0.5) + pt.x;

    int img_y = (int)(pt.y + 0.5);
    int img_x = (int)(pt.x + 0.5);
    return sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1) - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL) - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1) + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
}

// Remove all keypoints which the corresponding 3D point is (nan, nan, nan) and (0,0,0)
void remove_keypoints_on_holes(const pcl::PolygonMesh &pcl_mesh, std::vector<cv::KeyPoint> &keypoints, float scale)
{
    CloudType cloud;
    pcl::fromPCLPointCloud2(pcl_mesh.cloud, cloud);

    int num_vertices = cloud.size();
    int num_faces = pcl_mesh.polygons.size();

    std::vector<cv::KeyPoint> valid_keypoints;
    for (int i = 0; i < keypoints.size(); ++i)
    {
        int img_x = (int)(keypoints[i].pt.x + 0.5) * scale;
        int img_y = (int)(keypoints[i].pt.y + 0.5) * scale;
        if (!pcl::isFinite(cloud.at(img_x, img_y)) ||
            (cloud.at(img_x, img_y).x == 0 && cloud.at(img_x, img_y).y == 0 && cloud.at(img_x, img_y).z == 0))
            continue;

        valid_keypoints.push_back(keypoints[i]);
        //std::cout << cloud.at(img_x, img_y) << endl;
    }
}

void pclMesh2libgeodesicMesh(const pcl::PolygonMesh &pcl_mesh,
                             hmTriDistance *distance,
                             hmTriMesh *mesh,
                             std::vector<int> &inverse_shift_connected_vertices,
                             std::vector<int> &shifts)
{
    hmTriMeshDestroy(mesh);
    hmTriMeshInitialize(mesh);

    CloudType cloud;
    pcl::fromPCLPointCloud2(pcl_mesh.cloud, cloud);

    int num_vertices = cloud.size();
    int num_faces = pcl_mesh.polygons.size();
    // Look for all connected vertices, if a vertice is connected then 0 otherwise 1

    std::cout << "#Face: " << num_faces << std::endl
              << std::flush;

    std::cout << "Finding unconnected vertices\n"
              << std::flush;

    std::vector<int> connected_vertices(num_vertices, 1);
    for (int f_idx = 0; f_idx < num_faces; ++f_idx)
        for (int v_idx = 0; v_idx < 3; ++v_idx)
            connected_vertices[pcl_mesh.polygons[f_idx].vertices[v_idx]] = 0;

    // It will be used to access the removed vertices
    int num_unconnected_vertices = std::accumulate(connected_vertices.begin(),
                                                   connected_vertices.end(), 0);
    inverse_shift_connected_vertices.resize(num_vertices - num_unconnected_vertices, 0);

    std::cout << "#Unconnected vertices: " << num_unconnected_vertices << std::endl
              << std::flush;

    std::cout << "Computing shifts" << std::endl
              << std::flush;

    shifts.resize(num_vertices, 0);
    shifts[0] = connected_vertices[0];
    for (int v_idx = 1; v_idx < shifts.size(); ++v_idx)
        shifts[v_idx] = shifts[v_idx - 1] + connected_vertices[v_idx];

    std::cout << "Updating indices of connected vertices and loading faces\n"
              << std::flush;

    mesh->nFaces = num_faces;
    mesh->nVertices = inverse_shift_connected_vertices.size();

    /* allocate storage */
    mesh->vertices = (double *)malloc(mesh->nVertices * 3 * sizeof(double));
    mesh->texCoords = (double *)malloc(mesh->nVertices * 2 * sizeof(double));
    mesh->faces = (size_t *)malloc(mesh->nFaces * 3 * sizeof(size_t));

    size_t *f = mesh->faces;

    // Update the indices for all connected vertices

    for (int f_idx = 0; f_idx < num_faces; ++f_idx)
    {
        for (int v_idx = 0; v_idx < 3; ++v_idx)
        {
            size_t index = pcl_mesh.polygons[f_idx].vertices[v_idx];
            size_t new_index = index - shifts[index];

            inverse_shift_connected_vertices[new_index] = shifts[index];

            f[v_idx] = new_index;
        }
        f += 3;
    }

    std::cout << "Removing unconnected vertices and loading them\n"
              << std::flush;

    double *v = mesh->vertices;
    for (int v_idx = 0; v_idx < num_vertices; ++v_idx)
    {
        // Include connected vertices only
        if (connected_vertices[v_idx] == 1)
            continue;

        v[0] = cloud[v_idx].x;
        v[1] = cloud[v_idx].y;
        v[2] = cloud[v_idx].z;
        v += 3;
    }

    distance->surface = mesh;
}

bool setSources(hmTriDistance *distance, size_t keypoint_index)
{
    size_t i, n;
    size_t nVertices = distance->surface->nVertices;

    /* initialize all vertices to zero, meaning "not a source" */
    hmClearArrayDouble(distance->isSource.values, nVertices, 0.);

    /* set the specified source vertices in the current set */

    /* make sure the source vertex index n is valid */
    n = keypoint_index;
    if (n >= nVertices)
    {
        /* print an error message, remembering that source
      * vertices were 1-based in the input */
        PCL_ERROR("Error: source vertices must be in the range 1-nVertices!\n");
        return false;
    }

    /* flag the current vertex as a source */
    distance->isSource.values[n] = 1.;
    return true;
}

std::vector<double> interpolate_heatflow(std::vector<double> &heatflow, float scale, int width, cv::Point2f p)
{
    int height = heatflow.size() / width;
    cv::Mat heatflow_img(height, width, CV_64FC1);
    cv::Point2f shift_error;

    p.x = (int)(p.x + 0.5);
    p.y = (int)(p.y + 0.5);

    for (size_t i = 0; i < heatflow.size(); i++)
        heatflow_img.at<double>(i / width, i % width) = heatflow[i];

    cv::Mat hscaled_off, hscaled;
    cv::resize(heatflow_img, hscaled_off, /*cv::Size(640,480)*/ cv::Size(), 1.f / scale, 1.f / scale, CV_INTER_LINEAR);

    //printf("Interpolated heatflow (%d, %d)\n", hscaled_off.cols, hscaled_off.rows);

    shift_error.x = p.x - ((int)(p.x * scale)) / scale;
    shift_error.y = p.y - ((int)(p.y * scale)) / scale;

    //cout<<"Erro X: " << shift_error.x << " Erro Y: " << shift_error.y << endl;

    if (scale != 1.0)
        apply_offset2d(hscaled_off, hscaled, shift_error.x - 1, shift_error.y - 1);
    else
        hscaled = hscaled_off;

    cv::Mat outimg;
    char buf[256];


    std::vector<double> new_heatflow;

    for (size_t i = 0; i < hscaled.rows * hscaled.cols; i++)
        new_heatflow.push_back(hscaled.at<double>(i / hscaled.cols, i % hscaled.cols));

    return new_heatflow;
}

// Rotate a 2D point in counter clockwise rotation
// angle in radians
cv::Point2f rotate2d(const cv::Point2f &inPoint, const float &angRad)
{
    cv::Point2f outPoint;

    outPoint.x = std::cos(angRad) * inPoint.x - std::sin(angRad) * inPoint.y;
    outPoint.y = std::sin(angRad) * inPoint.x + std::cos(angRad) * inPoint.y;

    return outPoint;
}

void apply_offset2d(cv::Mat &in, cv::Mat &out, int offsetx, int offsety)
{
    out = in.clone();
    cv::Rect rsource = cv::Rect(cv::max(0, -offsetx), cv::max(0, -offsety), in.cols - abs(offsetx), in.rows - abs(offsety));
    cv::Rect rtarget = cv::Rect(cv::max(0, offsetx), cv::max(0, offsety), in.cols - abs(offsetx), in.rows - abs(offsety));

    in(rsource).copyTo(out(rtarget));
}

void compute_vector_feature(const CloudType::Ptr cloud, std::string img_path, const std::vector<std::vector<double>> &distance,
                            std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat &descriptors,
                            std::string sourcedir,
                            std::string dataset_type,
                            int patchSize,
                            float max_isocurve_size,
                            std::string use_kp_orientation)
{

    int halfPatchSize = patchSize / 2;
    cv::Mat img = cv::imread(img_path.c_str());;
    //extract_image_from_pointcloud(cloud, img, img_path, dataset_type);

    //const std::string test_pairs_file = sourcedir + "/aux/test_pairs_reaching_holes.txt"; // "test_pairs.txt";
    //const std::string test_pairs_file = sourcedir + "/aux/test_pairs_512.txt";
    const std::string test_pairs_file = sourcedir + "/aux/test_pairs_1024.txt";
    std::vector<std::vector<float>> test_pairs;

    load_test_pairs(test_pairs_file, test_pairs);

    int num_bytes = test_pairs.size() / 8;
    //descriptors = cv::Mat::zeros(keypoints.size(), num_bytes, CV_8U);

    cv::Mat grayImage = img;
    if (img.type() != CV_8U)
        cvtColor(img, grayImage, CV_BGR2GRAY);

    // Construct integral image for fast smoothing (box filter)
    cv::Mat sum;
    cv::integral(grayImage, sum, CV_32S);

    std::vector<cv::KeyPoint> valid_keypoints;

    // pre-compute the end of a row in a circular patch (used for canonicall orientation)
    std::vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    //cv::namedWindow("test_pairs", cv::WINDOW_NORMAL);
    for (size_t kp = 0; kp < keypoints.size(); ++kp)
    {
        //std::cout << std::endl << "Handling keypoint # " << kp << " " << keypoints[kp].pt << std::endl << std::flush;

        cv::Mat img_distance(grayImage.rows, grayImage.cols, CV_64FC1);

        for (int d = 0; d < distance[kp].size(); ++d)
        {
            double dist = distance[kp][d];

            //if (dist < 0)
            //    dist = std::numeric_limits<double>::quiet_NaN();

            int x = d % grayImage.cols;
            int y = d / grayImage.cols;

            img_distance.at<double>(y, x) = dist;
        }

        float canonical_angle = compute_canonical_orientation(grayImage, keypoints[kp], umax, patchSize, use_kp_orientation);

        //cv::Mat heat_flow_colormap;
        //func_display(distance[kp], cloud, heat_flow_colormap);

        // Keypoint on a hole (there is no geodesic info for it)
        cv::KeyPoint _keypoint = keypoints[kp];
        _keypoint.pt.x = (int)(keypoints[kp].pt.x + 0.5);
        _keypoint.pt.y = (int)(keypoints[kp].pt.y + 0.5);

        if ( //img_distance.at<double>(_keypoint.pt) > 0.02 ||
            //std::fpclassify(img_distance.at<double>(_keypoint.pt)) != FP_ZERO ||
            img_distance.at<double>(_keypoint.pt) < 0)
        {
            printf("Keypoint with invalid distance: %f!\n", img_distance.at<double>(_keypoint.pt));
            std::cout << "Invalid distance: " << _keypoint.pt << "\t\t\t "
                      //<< cloud->at(_keypoint.pt.x, _keypoint.pt.y)
                      << "\t\tDistance: " << img_distance.at<double>(_keypoint.pt) << std::endl
                      << std::flush;

            //show_keypoint_on_heatflow(cloud, distance[kp], keypoints[kp]);
            cv::circle(img, _keypoint.pt, 3, cv::Scalar(0, 0, 0), 1);

            continue;
        }

        valid_keypoints.push_back(_keypoint);

        cv::Mat keypoint_descriptor = cv::Mat::zeros(1, num_bytes, CV_8U);
        uchar *desc = keypoint_descriptor.ptr(0);

        char bitstring[4096] = "";
        uchar feat = 0;

        for (size_t tp = 0; tp < test_pairs.size(); ++tp)
        {
            //std::cout << "=== Test pair # " << tp << "===" << std::endl << std::flush;

            float isocurve = test_pairs[tp][0];
            cv::Point2f pointX;
            float dir = canonical_angle + test_pairs[tp][1];
            estimatePointAtIsoCurve(img_distance, _keypoint, dir, isocurve, pointX, max_isocurve_size);
            //cv::circle(img, pointX, 2, cv::Scalar(0, 0, 255));
            //cv::circle(heat_flow_colormap, pointX, 2, cv::Scalar(0, 0, 0), 1);

            //std::cout << "PointX: " << pointX << std::endl
            //    << "\t" << test_pairs[tp][1] << " " << isocurve << std::endl
            //    << std::flush;

            isocurve = test_pairs[tp][2];
            cv::Point2f pointY;
            dir = canonical_angle + test_pairs[tp][3];
            estimatePointAtIsoCurve(img_distance, _keypoint, dir, isocurve, pointY, max_isocurve_size);

            uchar vtest = smoothedSum(sum, pointX) < smoothedSum(sum, pointY);
            //bool gtest = abs(img_distance.at<float>(pointX) - img_distance.at<float>(pointY)) > 0.001;

            uchar test = vtest; //| gtest;

            sprintf(bitstring, "%s %d", bitstring, test);
            feat = feat + (test << (tp % 8));
            //std::cout << std::endl << (int)feat;
            if ((tp + 1) % 8 == 0)
            {
                desc[tp / 8] = feat;
                feat = 0;
                //std::cout << std::endl << bitstring << " (" << (int)desc[tp/8] << ")"<< std::endl;
                //bitstring[0] = '\0';
                //getchar();
            }
        }
        //std::cout << descriptors.row(kp) << std::endl << std::flush;
        //std::cout << "\nDescriptor: [" << bitstring << "]" << std::endl << std::flush;

        descriptors.push_back(keypoint_descriptor);
    }

    keypoints.clear();
    keypoints = valid_keypoints;
    printf("Number of valid keypoints: %d\n", valid_keypoints.size());
    printf("Number of valid descriptors: %d\n", descriptors.rows);
}

std::vector<cv::DMatch> calcAndSaveHammingDistancesNonrigid(std::vector<cv::KeyPoint> kp_query,
                                                            std::vector<cv::KeyPoint> kp_tgt,
                                                            std::vector<cv::Mat> desc_query,
                                                            std::vector<cv::Mat> desc_tgt,
                                                            CSVTable query,
                                                            CSVTable tgt,
                                                            std::string file_name,
                                                            int nb_angle_bins)
{
    //We are going to create a matrix of distances from query to desc and save it to a file 'IMGNAMEREF_IMGNAMETARGET_DESCRIPTORNAME.txt'
    std::vector<cv::DMatch> matches;

    std::ofstream oFile(file_name.c_str());

    oFile << query.size() << " " << tgt.size() << std::endl;

    cv::Mat dist_mat(query.size(), tgt.size(), CV_32S, cv::Scalar(-1));

    int c_hits = 0;

    for (size_t i = 0; i < desc_query[0].rows; i++)
    {
        int menor = 999, menor_idx = -1, menor_i = -1, menor_j = -1;

        for (size_t j = 0; j < desc_tgt[0].rows; j++)
        {
            int _i = kp_query[i].class_id; //correct idx
            int _j = kp_tgt[j].class_id;   //correct idx

            //if(_i < 0 || _i >= dist_mat.rows || _j < 0 || _j >= dist_mat.cols)
            //    std::cout << "Estouro: " << _i << " " << _j << std::endl;

            if (!(query[_i]["valid"] == 1 && tgt[_i]["valid"] == 1)) //this match does not exist
                continue;

            if (query[_i]["valid"] == 1 && tgt[_j]["valid"] == 1)
            {

                dist_mat.at<int>(_i, _j) = norm_hamming_nonrigid(desc_query, desc_tgt, i, j, nb_angle_bins);
                if (dist_mat.at<int>(_i, _j) < menor)
                {
                    menor = dist_mat.at<int>(_i, _j);
                    menor_i = _i;
                    menor_j = _j;
                    menor_idx = j;
                }
            }

            //oFile << cv::norm(desc_query.row(i), desc_tgt.row(j),cv::NORM_HAMMING) << " ";
        }

        cv::DMatch d;
        d.distance = menor;
        d.queryIdx = i;
        d.trainIdx = menor_idx;

        if (d.queryIdx >= 0 && d.trainIdx >= 0)
        {
            matches.push_back(d);
            if (menor_i == menor_j)
                c_hits++;
        }
    }

    for (int i = 0; i < dist_mat.rows; i++)
        for (int j = 0; j < dist_mat.cols; j++)
        {
            oFile << dist_mat.at<int>(i, j) << " ";
        }

    oFile << std::endl;
    oFile.close();
    std::cout << "Correct matches: " << c_hits << " of " << matches.size() << std::endl;

    return matches;
}

int norm_hamming_nonrigid(std::vector<cv::Mat>& src, std::vector<cv::Mat>& tgt, int idx_d1, int idx_d2, int nb_angle_bins)
{
    std::vector<int> distances;

    for(int i=0; i < nb_angle_bins; i++)
        distances.push_back(cv::norm(src[0].row(idx_d1), tgt[i].row(idx_d2),cv::NORM_HAMMING));

    size_t min_idx =  std::distance(std::begin(distances),std::min_element(std::begin(distances), std::end(distances)));
        
    return distances[min_idx];
    
}

void filter_matches(const std::vector<cv::DMatch> &matches, 
                    int threshold, std::vector<cv::DMatch> &filtered_matches)
{

    //std::cout<<"Matches before filtering: " << matches.size() << std::endl;
    for (size_t m = 0; m < matches.size(); ++m)
    {
        if (matches[m].distance < threshold)
            filtered_matches.push_back(matches[m]);
    }
    //std::cout<<"Matches after filtering: " << filtered_matches.size() << std::endl;


}

 std::vector<cv::DMatch> validate_matches(std::vector<cv::KeyPoint>& src, std::vector<cv::KeyPoint>& dst, std::vector<cv::DMatch> &matches, cv::Mat& img, int src_img_width)
 {
    std::vector<cv::DMatch> valid_ones;

    for(size_t i=0; i < matches.size(); i++)
    {
        cv::KeyPoint kp1 = src[matches[i].queryIdx];
        cv::KeyPoint kp2 = dst[matches[i].trainIdx];

        if(kp1.class_id == kp2.class_id)
        {
            valid_ones.push_back(matches[i]);
            cv::line(img,kp1.pt,cv::Point2f(kp2.pt.x + src_img_width,kp2.pt.y),cv::Scalar(0,255,0),1,CV_AA);
        }

    }

    return valid_ones;
 }
 

 std::vector<cv::DMatch> match_and_filter(std::vector< cv::Mat > d1,  std::vector < cv::Mat > d2, std::vector<cv::KeyPoint> k1, std::vector<cv::KeyPoint> k2, std::string out_filename,int nb_angle_bins)
{
    std::vector<cv::DMatch> matches;
    std::vector<int> distances(d2[0].rows);
    std::ofstream oFile(out_filename.c_str());
    cout << "saving in " << out_filename << endl;

    for(size_t i=0; i <d1[0].rows; i++)
    {   
        for(size_t j = 0; j < d2[0].rows; j++)
          distances[j] = norm_hamming_nonrigid(d1, d2, i, j,  nb_angle_bins); 

      size_t min_idx =  std::distance(std::begin(distances),std::min_element(std::begin(distances), std::end(distances)));
      float nn_val = distances[min_idx];

      cv::Point2f p1, p2;
      p1 = k1[i].pt;
      p2 = k2[min_idx].pt;
        cv::DMatch d;
        d.queryIdx = i;
        d.trainIdx = min_idx;

      distances[min_idx] = 1024;
      min_idx = std::distance(std::begin(distances),std::min_element(std::begin(distances), std::end(distances)));
      float nn2_val = distances[min_idx];

      //if(0.9*nn2_val > nn_val) //good match
      //{

        matches.push_back(d);
        oFile << p1.x <<" " << p1.y << " " << p2.x << " " << p2.y << std::endl;
      //}

    }   

    oFile.close();
    return matches;
}

std::vector<cv::DMatch> calcAndSaveDistances(std::vector<cv::KeyPoint> kp_query, std::vector<cv::KeyPoint> kp_tgt, 
cv::Mat desc_query, cv::Mat desc_tgt, CSVTable query, CSVTable tgt, std::string file_name, int normType)
 {
   //We are going to create a matrix of distances from query to desc and save it to a file 'IMGNAMEREF_IMGNAMETARGET_DESCRIPTORNAME.txt'
   std::vector<cv::DMatch> matches;
   
   std::ofstream oFile(file_name.c_str());
   
   oFile << query.size() << " " << tgt.size() << std::endl;
   
   cv::Mat dist_mat(query.size(),tgt.size(),CV_64F,cv::Scalar(-1));
   
    int c_hits=0;

   for(size_t i=0; i < desc_query.rows; i++)
   {
    double menor = 99999;
    size_t menor_idx=-1, menor_i=-1, menor_j = -1;
     
    for(size_t j = 0; j < desc_tgt.rows; j++)
      {
        int _i = kp_query[i].class_id; //correct idx
        int _j = kp_tgt[j].class_id; //correct idx
        
        //if(_i < 0 || _i >= dist_mat.rows || _j < 0 || _j >= dist_mat.cols)
        //    std::cout << "Estouro: " << _i << " " << _j << std::endl;
        
        if(!(query[_i]["valid"] == 1 && tgt[_i]["valid"] == 1)) //this match does not exist
          continue;

        if(query[_i]["valid"] == 1 && tgt[_j]["valid"] == 1)
        {
            dist_mat.at<double>(_i,_j) = cv::norm(desc_query.row(i), desc_tgt.row(j),normType);

          if(dist_mat.at<double>(_i,_j) < menor )
          {
                        menor = dist_mat.at<double>(_i,_j);
                        menor_i = _i;
                        menor_j = _j;
                        menor_idx = j;
                    }
        }
        
        //oFile << cv::norm(desc_query.row(i), desc_tgt.row(j),cv::NORM_HAMMING) << " ";
      }

          cv::DMatch d;
          d.distance = menor;
          d.queryIdx = i;
          d.trainIdx = menor_idx;

        if(d.queryIdx >=0 && d.trainIdx >=0)
        {
            matches.push_back(d);
            if(menor_i == menor_j)
                c_hits++;
        }

    }
      
  for(int i=0; i < dist_mat.rows;i++)
    for(int j=0; j < dist_mat.cols; j++)
    {
      oFile << dist_mat.at<double>(i,j) << " ";
    }
    
    oFile << std::endl;   
    oFile.close(); 
    std::cout <<"Correct matches: " << c_hits << " of " << matches.size() << std::endl;
   
   
   return matches;
 }
