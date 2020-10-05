#include <iostream>
#include <string>
#include <sstream>
#include <numeric>      // std::accumulate
#include <algorithm>    // std::min_element, std::max_element
#include <map>
#include <chrono>
#include <stdio.h>

#include <math.h>       // for M_PI

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>

#include <pcl/surface/organized_fast_mesh.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>

// OpenCV
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>


// Matlab
//#include "engine.h" 

// libgeodesic
#include "hmTriDistance.h"
#include "hmContext.h"
#include "hmUtility.h"
#include "hmVectorSizeT.h"

typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointCloud<PointType> CloudType;
typedef std::vector<std::map<std::string,float> > CSVTable;

typedef std::vector<std::vector<double> > vec2d;

//#define DEBUG_ORIENTATION
#define USE_ROTATED_PATTERNS
#define SHOW_TEST_PAIRS 0
#define USE_KEYPOINT_ORIENTATION "ORB" // use "" to disable, "ORB" or "SURF" for ORB or SURF orientation estimation
#define RECALC_HEATFLOW 1

// Opencv 4 compatibility 
#ifdef CV4

#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_INTER_LINEAR cv::INTER_LINEAR
#define CV_AA cv::LINE_AA

#endif


long int c_keypoints =0;
bool use_detector = false;

// Descriptor parameters
static const int patchSize = 31;            // canonical orientation //int patchSize = 31;  // default value used in ORB // 31 - 15 - 7
int halfPatchSize = patchSize / 2;
int nb_angle_bins = 12;

static const int KERNEL_SIZE = 9;           // sum smoothing // defaul value used in BRIEF: KERNEL_SIZE = 9

float max_isocurve_size;    // thickness of each isocurve
float keypoint_scale = 7.0; //default OpenCV KeyPoint.size for the FAST detector

int pyramid_levels = 0;

std::string dataset_type;
std::string sourcedir;

cv::Mat global_K, global_Kinv;

//Time measuring variables
double heatflow_timesum=0;
double binary_extraction=0;
double matching_sum=0;


inline int coords2index(const cv::KeyPoint &keypoint, int width){
    return keypoint.pt.y * width + keypoint.pt.x;
}

int norm_hamming_nonrigid(std::vector<cv::Mat>& src, std::vector<cv::Mat>& tgt, int idx_d1, int idx_d2);

// Compute the canonical orientation based on the Intensity Centroid approach
// Return the angle in radian
float compute_canonical_orientation(const cv::Mat &image, cv::KeyPoint &keypoint, 
                                    const std::vector<int> &umax )
{
// extracted from OpenCV (ORB.cpp and features2d.hpp)
// static float IC_Angle(const Mat& image, const int half_k, Point2f pt, const vector<int> & u_max)

    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar>(cvRound(keypoint.pt.y), cvRound(keypoint.pt.x));

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
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    float orientation = 0; 
    keypoint.angle = -1;

    if(!strcmp(USE_KEYPOINT_ORIENTATION, "ORB"))
    {
        orientation = (cv::fastAtan2((float)m_01, (float)m_10) * (float)(CV_PI/180.f)); //orientation used by the non_rigid_desc
        keypoint.angle = cv::fastAtan2((float)m_01, (float)m_10); //orientation used by other descriptors
    }
    //
    //keypoint.angle = cv::fastAtan2((float)m_01, (float)m_10);
    //return orientation;
    else if (!strcmp(USE_KEYPOINT_ORIENTATION, "SURF"))
   {
    //cv::Ptr<cv::Feature2D> surf = cv::SURF::create();//cv::Algorithm::create<cv::Feature2D>("Feature2D.SURF");
    //if (surf.empty())
    //    CV_Error(CV_StsNotImplemented, "OpenCV was built without SURF support.");

    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();// surf = cv::DescriptorExtractor::create("SURF");
    std::vector<cv::KeyPoint> keypoints;
    keypoints.push_back(keypoint);
    cv::Mat d;
    //std::cout << "**angle " << keypoint.pt << image.rows << " " << image.cols << std::endl;
    surf->compute(image,keypoints,d);
    //surf->operator()(image, cv::noArray(), keypoints, cv::noArray(), true);

    orientation = keypoints[0].angle * (float)(CV_PI/180.f);
    keypoint.angle = keypoints[0].angle;
    
   // std::cout << "**angle " << keypoint.angle << std::endl;
    }

    return orientation;
}


cv::Mat nanConv(cv::Mat img)
{    
    cv::Mat kernelX = cv::getGaussianKernel(7, 1.1, CV_64F );
    cv::Mat kernelY = cv::getGaussianKernel(7, 1.1, CV_64F );
    cv::Mat kernel = kernelX * kernelY.t();

    cv::Mat mat_ones = cv::Mat::ones(img.rows, img.cols, CV_64F);
    cv::Mat mat_conv = img.clone();

    for(int i=0; i < mat_conv.rows; i++)
        for(int j=0; j < mat_conv.cols; j++)
        {
             if( std::isnan(img.at<double>(i,j)) || img.at<double>(i, j) == 0)
             {
                mat_conv.at<double>(i,j) = 0;
                mat_ones.at<double>(i,j) = 0;
             }

        }

    cv::Mat conv1;
    cv::Mat conv2;
    cv::Mat smoothed;

    cv::filter2D(mat_conv, conv1, -1, kernel, cv::Point(-1,-1), 0,cv::BORDER_DEFAULT);
    cv::filter2D(mat_ones, conv2, -1, kernel, cv::Point(-1,-1), 0,cv::BORDER_DEFAULT);

    //conv1.setTo(0, conv1 <= 0);
    //conv2.setTo(0, conv2 <= 0);

    smoothed = conv1 / conv2;

    return smoothed;
}

cv::Mat pyramid_downsample(cv::Mat image, int levels)
{
    cv::Mat smoothed = image.clone();

    for(int l =0; l < levels; l++)
    {
         smoothed = nanConv(smoothed);
         cv::resize(smoothed, smoothed, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

    }

    return smoothed;
}

void loadCloudFromPNGImages(const std::string &inputdir, const std::string &filename, CloudType::Ptr cloud, cv::Mat &rgb)
{

    printf("Loading cloud from %s/%s\n", inputdir.c_str(), filename.c_str());
    cv::FileStorage fs;
    cv::Mat in_depth, depth, matcloud;
    cv::Mat K;

    fs.open(inputdir + "/intrinsics.xml", cv::FileStorage::READ);
    fs["intrinsics"] >> K; 
    std::cout << K << std::endl; 
    global_K = K.clone();
    global_K.convertTo(global_K, CV_64F);
    global_Kinv = global_K.inv();

    rgb = cv::imread(inputdir + "/" + filename + "-rgb.png");
    in_depth = cv::imread(inputdir + "/" + filename + "-depth.png", cv::IMREAD_ANYDEPTH);
    in_depth.convertTo(in_depth, CV_64F);
    in_depth = in_depth / 1000.0;

    
    //cloud_mats[filename] = createCloudCvMat(in_depth, K);

    K = K / pow(2, pyramid_levels);
    K.convertTo(K,CV_64F);
    K.at<double>(2,2) = 1.0;
    cv::Mat Kinv = K.inv();

    //cv::Mat out = in_depth * 50.0;
    //out.convertTo(out, CV_8U);
    //cv::imwrite("full.png", out); 
    //apply pyramid smoothing here
    //std::cout << K.inv() << std::endl; getchar();
    depth = pyramid_downsample(in_depth, pyramid_levels);
    //out = depth * 50.0;
    //out.convertTo(out, CV_8U);
    //cv::imwrite("smoothed.png", out); printf("wrote\n"); exit(0);

    // populate our PointCloud with points
    cloud->width    = depth.cols;
    cloud->height   = depth.rows;
    cloud->is_dense = true;
    cloud->points.resize (cloud->width * cloud->height);

    for (int y = 0 ; y < cloud->height; y++)
        for(int x = 0 ; x < cloud->width ; x++)
        {
            PointType point;
            //point = (*cloud)(x,y);
            cv::Mat pnorm = (cv::Mat_<double>(3,1) << x, y, 1.0);
            pnorm = Kinv * pnorm;
            double xn = pnorm.at<double>(0,0), yn = pnorm.at<double>(1,0);
            double x3d, y3d, z3d;
            z3d = depth.at<double>(y,x);

            if(z3d != 0)
            {
                x3d= xn * z3d;
                y3d= yn * z3d;
            }
            else
                x3d = y3d = z3d = std::numeric_limits<double>::quiet_NaN();

            point.x = x3d; point.y = y3d; point.z = z3d;
            
            point.b = rgb.at<cv::Vec3b>(y,x)[0];
            point.g = rgb.at<cv::Vec3b>(y,x)[1];
            point.r = rgb.at<cv::Vec3b>(y,x)[2];
            (*cloud)(x,y) = point;

            //printf("%.2f %.2f %.2f\n", x3d, y3d, z3d);
        }

    printf("Loaded cloud: %d x %d\n", cloud->width, cloud->height);
}

void func_display(const std::vector<double> &distance, cv::Mat img, cv::Mat &colormap) {

    cv::Mat img_distance(img.rows, img.cols, CV_64FC1);

    for (int d = 0; d < distance.size(); ++d)
    {
        double dist = distance[d];

        if (dist < 0)
            dist = std::numeric_limits<double>::quiet_NaN();

        int x = d % img.cols;
        int y = d / img.cols;

        float p; //45
        if(dataset_type == "realdata-smoothed" || dataset_type == "realdata-standard")
            p = 45;
        else
            p = 4.5;
        
        img_distance.at<double>(y, x) = cos(2 * M_PI * p * dist);

        //img_distance.at<double>(y, x) = distance[d];
        
    }

    //for(int i=0; i< 15; i++ )
     //   printf("%.3f ",img_distance.at<double>(240+i, 320+i));

    cv::Mat normalized_image_distance;
    cv::normalize(img_distance, normalized_image_distance, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(normalized_image_distance, colormap, cv::COLORMAP_AUTUMN);


    float alpha = 0.5;

    for(int y = 0; y < img.rows ; y++)
        for(int x=0; x < img.cols; x++)
        {
            colormap.at<cv::Vec3b>(y,x)[2] = (unsigned char)((float)colormap.at<cv::Vec3b>(y,x)[2]*(1.0-alpha) + (float)img.at<cv::Vec3b>(y,x)[2]*alpha);
            colormap.at<cv::Vec3b>(y,x)[1] = (unsigned char)((float)colormap.at<cv::Vec3b>(y,x)[1]*(1.0-alpha) + (float)img.at<cv::Vec3b>(y,x)[1]*alpha);
            colormap.at<cv::Vec3b>(y,x)[0] = (unsigned char)((float)colormap.at<cv::Vec3b>(y,x)[0]*(1.0-alpha) + (float)img.at<cv::Vec3b>(y,x)[0]*alpha);
        }
        


}

// Rotate a 2D point in counter clockwise rotation
// angle in radians
cv::Point2f rotate2d(const cv::Point2f& inPoint, const float& angRad)
{
    cv::Point2f outPoint;
    
    outPoint.x =  std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y =  std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;

    return outPoint;
}

void smooth_pcl(const CloudType::Ptr cloud, double radius)
{

  // Create a KD-Tree
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);

  // Output has the PointNormal type in order to store the normals calculated by MLS
  pcl::PointCloud<pcl::PointNormal> mls_points;

  // Init object (second point type is for the normals, even if unused)
  pcl::MovingLeastSquares<PointType, pcl::PointNormal> mls;
 
  //mls.setComputeNormals (true);

  // Set parameters
  mls.setInputCloud (cloud);
  mls.setPolynomialOrder (2);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (radius);

  // Reconstruct
  mls.process (mls_points);
}

// Extract rgb data from a PointCloud<PointRGB>
void extract_image_from_pointcloud(const CloudType::Ptr cloud, cv::Mat &rgb, std::string img_path)
{
    

    if(dataset_type == "realdata-smoothed" || dataset_type == "synthetic-smoothed")
        rgb = cv::imread(img_path.c_str());

    else
    {

        rgb = cv::Mat(cloud->height, cloud->width, CV_8UC3);

        for (int c = 0; c < cloud->width; ++c)
            for (int r = 0; r < cloud->height; ++r)
            {
                rgb.at<cv::Vec3b>(r, c).val[2] = cloud->at(c, r).r;
                rgb.at<cv::Vec3b>(r, c).val[1] = cloud->at(c, r).g;
                rgb.at<cv::Vec3b>(r, c).val[0] = cloud->at(c, r).b;
            }
        }
}

// Extract rgb data from a PointCloud<PointRGB>
void extract_nonholesmask_from_pointcloud(const CloudType::Ptr cloud, cv::Mat &mask)
{
    double epsilon = 0.00001;
    float dthreshold = 0.2;
    int slide = 4;

    mask = cv::Mat::ones(cloud->height, cloud->width, CV_8UC1)*255; 

    for (int c = 0; c < cloud->width; ++c)
        for (int r = 0; r < cloud->height; ++r) {
            if (pcl_isnan(cloud->at(c, r).z) || !pcl::isFinite(cloud->at(c, r)) || cloud->at(c, r).z > 40.0 /*2*/) /* ||
                (abs(cloud->at(c, r).x) < epsilon && abs(cloud->at(c, r).y) < epsilon && abs(cloud->at(c, r).z) < epsilon) )*/
                mask.at<char>(r, c) = 0;
            /*
            else if ((cloud->at(c, r).z/cloud->at(c, r + slide).z > dthreshold) || 
                     (cloud->at(c, r).z/cloud->at(c, r - slide).z > dthreshold) ||
                     (cloud->at(c, r).z/cloud->at(c + slide, r).z > dthreshold) || 
                     (cloud->at(c, r).z/cloud->at(c - slide, r).z > dthreshold))
                mask.at<char>(r, c) = 0;
                */
        }
}

std::string dirnameOf(const std::string& fname)
{
     size_t pos = fname.find_last_of("\\/");
     return (std::string::npos == pos)
         ? ""
         : fname.substr(0, pos);
}

void load_test_pairs(const std::string &filename, std::vector< std::vector<float> > &test_pairs) 
{
    std::ifstream fin(filename.c_str());
    if (fin.is_open())
    {
        while(!fin.eof())
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

            if(!fin.eof())
                test_pairs.push_back(r_and_theta);
        }
        fin.close();
        
        std::cout <<"Loaded " << test_pairs.size() << " pair tests from the file." << std::endl;
    }
    else
    {
        PCL_ERROR("ERROR LOADING THE PAIRS TXT FILE...\n");
    }
}

void show_keypoint_on_heatflow(cv::Mat img, const std::vector<double> &dist_heat_flow, const cv::KeyPoint &kp)
{
    c_keypoints++;

    if(false && c_keypoints%15==0)
    {
    cv::Mat heat_flow;
    func_display(dist_heat_flow, img, heat_flow);

    cv::circle(heat_flow, cv::Point2f((int)(kp.pt.x+0.5),(int)(kp.pt.y+0.5)), 1, cv::Scalar(0, 0, 0));

    //cv::imshow("jet colormap", heat_flow);
    char buffer[256];
    sprintf(buffer, "keypoint_on_heatflow_%d.png",c_keypoints);
    cv::imwrite(buffer, heat_flow);
    //cv::waitKey();

    }
}

void show_heatflow_on_mesh(const pcl::PolygonMesh &mesh, const std::vector<double> &dist_heat_flow)
{
#if 0    
    CloudType::Ptr cloud(new CloudType);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
    
    CloudType::Ptr color_cloud(new CloudType);
    pcl::copyPointCloud(*cloud, *color_cloud);

    cv::Mat heat_flow;
    func_display(dist_heat_flow, color_cloud, heat_flow);

    //cv::imshow("jet colormap", heat_flow);
    cv::imwrite("heatflow_mesh.png", heat_flow);
    //cv::waitKey();


    // used to show mesh and point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.3, 0.3, 0.3);
    
    cv::Mat rgb_image;
    extract_image_from_pointcloud(cloud, rgb_image);

    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(color_cloud);
    viewer->addPolygonMesh<PointType>(color_cloud, mesh.polygons, "mesh", 0);
    //pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
    viewer->addPointCloud<PointType>(cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");

    //viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 0.05, "normals");
    viewer->addCoordinateSystem(0.3);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
#endif 
}

void pclMesh2libgeodesicMesh(   const pcl::PolygonMesh &pcl_mesh, 
                                hmTriDistance* distance,
                                hmTriMesh* mesh, 
                                std::vector<int> &inverse_shift_connected_vertices,
                                std::vector<int> &shifts)
{
    hmTriMeshDestroy( mesh );
    hmTriMeshInitialize( mesh );
  
    CloudType cloud;
    pcl::fromPCLPointCloud2(pcl_mesh.cloud, cloud);
    
    int num_vertices = cloud.size();
    int num_faces = pcl_mesh.polygons.size();
    // Look for all connected vertices, if a vertice is connected then 0 otherwise 1

    std::cout << "#Face: " << num_faces << std::endl << std::flush;

    std::cout << "Finding unconnected vertices\n" << std::flush;

    std::vector<int> connected_vertices(num_vertices, 1);
    for (int f_idx = 0; f_idx < num_faces; ++f_idx)
        for (int v_idx = 0; v_idx < 3; ++v_idx)
            connected_vertices[pcl_mesh.polygons[f_idx].vertices[v_idx]] = 0;

    // It will be used to access the removed vertices
    int num_unconnected_vertices = std::accumulate( connected_vertices.begin(),
                                                    connected_vertices.end(), 0);
    inverse_shift_connected_vertices.resize(num_vertices-num_unconnected_vertices, 0);

    std::cout << "#Unconnected vertices: " << num_unconnected_vertices << std::endl << std::flush;


    std::cout << "Computing shifts" << std::endl << std::flush;

    shifts.resize(num_vertices, 0);
    shifts[0] = connected_vertices[0];
    for (int v_idx = 1; v_idx < shifts.size(); ++v_idx)
        shifts[v_idx] = shifts[v_idx - 1] + connected_vertices[v_idx];

    std::cout << "Updating indices of connected vertices and loading faces\n" << std::flush;

    mesh->nFaces = num_faces;
    mesh->nVertices = inverse_shift_connected_vertices.size();

    /* allocate storage */
    mesh->vertices  = (double*)malloc( mesh->nVertices*3 * sizeof(double) );
    mesh->texCoords = (double*)malloc( mesh->nVertices*2 * sizeof(double) );
    mesh->faces     = (size_t*)malloc(    mesh->nFaces*3 * sizeof(size_t) );

    size_t* f = mesh->faces;

    // Update the indices for all connected vertices

    for (int f_idx = 0; f_idx < num_faces; ++f_idx) {
        for (int v_idx = 0; v_idx < 3; ++v_idx)
        {               
            size_t index = pcl_mesh.polygons[f_idx].vertices[v_idx];
            size_t new_index = index - shifts[index];

            inverse_shift_connected_vertices[new_index] = shifts[index];

            f[v_idx] = new_index;
        }
        f += 3;
    }

    std::cout << "Removing unconnected vertices and loading them\n" << std::flush;

    double* v = mesh->vertices;
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

bool setSources( hmTriDistance* distance, size_t keypoint_index )
{
   size_t i, n;
   size_t nVertices = distance->surface->nVertices;

   /* initialize all vertices to zero, meaning "not a source" */
   hmClearArrayDouble( distance->isSource.values, nVertices, 0. );

   /* set the specified source vertices in the current set */

  /* make sure the source vertex index n is valid */
  n = keypoint_index;
  if( n >= nVertices )
  {
     /* print an error message, remembering that source
      * vertices were 1-based in the input */
     PCL_ERROR("Error: source vertices must be in the range 1-nVertices!\n" );
     return false;
  }

  /* flag the current vertex as a source */
  distance->isSource.values[n] = 1.;
  return true;
}

// Remove all keypoints which the corresponding 3D point is (nan, nan, nan) and (0,0,0)
void remove_keypoints_on_holes( const pcl::PolygonMesh &pcl_mesh, std::vector<cv::KeyPoint> &keypoints, float scale)
{  
    CloudType cloud;
    pcl::fromPCLPointCloud2(pcl_mesh.cloud, cloud);
    
    int num_vertices = cloud.size();
    int num_faces = pcl_mesh.polygons.size();

    std::vector<cv::KeyPoint> valid_keypoints;
    for(int i = 0; i < keypoints.size(); ++i)
    {
        int img_x = (int)(keypoints[i].pt.x + 0.5)*scale;
        int img_y = (int)(keypoints[i].pt.y + 0.5)*scale;
        if (!pcl::isFinite(cloud.at(img_x, img_y)) || 
            (cloud.at(img_x, img_y).x == 0 && cloud.at(img_x, img_y).y == 0 && cloud.at(img_x, img_y).z == 0))
            continue;
        
        valid_keypoints.push_back(keypoints[i]);
        //std::cout << cloud.at(img_x, img_y) << endl;
    }
}


bool load_heatflow_from_file(std::string filename, vec2d& heatflow)
{
    heatflow.clear();
    
    long k, wh;
    double d;
    
     FILE* pFile = fopen ( filename.c_str() , "rb" );
     
    if (pFile==NULL) 
    {
        //fputs ("File error",stderr); exit (1);
        return false;
    }

        fread(&k, sizeof(long), 1, pFile);
        fread(&wh, sizeof(long), 1, pFile);
        
        for(size_t i=0; i < k; i++)
        {
            std::vector<double> vec_d(wh);
           /* for(size_t j=0; j < wh; j++)
            {
                fread(&d, sizeof(double), 1, pFile);
                vec_d.push_back(d);
            }
            */
            fread(vec_d.data(), sizeof(double), vec_d.size(), pFile);
            
            heatflow.push_back(vec_d);
        }
        
     fclose(pFile);
     return true;
        
}

void dump_heatflow_to_file(std::string filename, vec2d heatflow)
{
    FILE * pFile;
    pFile = fopen (filename.c_str(), "wb");
    
    long k, wh;
    double d;
    
    int t;

    k = heatflow.size();
    wh = heatflow[0].size();
    
    fwrite (&k , sizeof(k), 1, pFile);
    fwrite (&wh , sizeof(wh), 1, pFile);

    for(size_t i=0; i < k; i++)
        for(size_t j=0; j < wh; j++)
            fwrite(&heatflow[i][j], sizeof(double), 1, pFile);
            
    fclose(pFile);
    
        
}

void apply_offset2d(cv::Mat& in, cv::Mat& out, int offsetx, int offsety)
{
    out = in.clone();
    cv::Rect rsource = cv::Rect(cv::max(0,-offsetx), cv::max(0,-offsety), in.cols-abs(offsetx), in.rows-abs(offsety));
    cv::Rect rtarget = cv::Rect(cv::max(0, offsetx), cv::max(0, offsety), in.cols-abs(offsetx), in.rows-abs(offsety));

    in(rsource).copyTo(out(rtarget));   
}

std::vector<double> interpolate_heatflow(std::vector<double>& heatflow, float scale, cv::Point2f p)
{
    cv::Mat heatflow_img(480.0*scale, 640.0*scale, CV_64FC1);
    int width = 640.0*scale;
    cv::Point2f shift_error;

    p.x = (int)(p.x + 0.5); p.y = (int)(p.y + 0.5);

    for(size_t i=0; i < heatflow.size(); i++)
        heatflow_img.at<double>(i/width, i%width) = heatflow[i];

    cv::Mat hscaled_off, hscaled;
    cv::resize(heatflow_img, hscaled_off, /*cv::Size(640,480)*/cv::Size(), 1.f/scale,1.f/scale, CV_INTER_LINEAR);

    //printf("Interpolated heatflow (%d, %d)\n", hscaled_off.cols, hscaled_off.rows);

    shift_error.x = p.x - ((int)(p.x*scale)) / scale;
    shift_error.y = p.y - ((int)(p.y*scale)) / scale;

    //cout<<"Erro X: " << shift_error.x << " Erro Y: " << shift_error.y << endl;
 
    if (scale != 1.0)
        apply_offset2d(hscaled_off, hscaled, shift_error.x-1, shift_error.y-1);
    else
        hscaled = hscaled_off;

    cv::Mat outimg;
    char buf[256];

    std::vector<double> new_heatflow;

    for(size_t i=0; i < hscaled.rows*hscaled.cols; i++)
        new_heatflow.push_back(hscaled.at<double>(i/640, i%640));

    return new_heatflow;


}

void compute_heat_flow_c(const pcl::PolygonMesh &pcl_mesh,
                        std::vector<cv::KeyPoint> &coords_keypoints, 
                        std::vector< std::vector<double> > &dist_heat_flow)
{
    
   /* main data */
   hmContext context;
   hmTriMesh surface;
   hmTriDistance distance;
   distance.verbose = 0;

   /* initialize data */
    //std::cout << "hmContextInitialize..." << std::endl << std::flush;        
   hmContextInitialize( &context );
    //std::cout << "hmTriMeshInitialize..." << std::endl << std::flush;        
   hmTriMeshInitialize( &surface );
   // std::cout << "hmTriDistanceInitialize..." << std::endl << std::flush;        
   hmTriDistanceInitialize( &distance );

   /* read surface */
    //std::cout << "pclMesh2libgeodesicMesh..." << std::endl << std::flush;        
    std::vector<int> inverse_shift_connected_vertices;
    std::vector<int> shifts;
    float f_scale = pcl_mesh.cloud.width/640.0;

    remove_keypoints_on_holes( pcl_mesh, coords_keypoints, f_scale );

    pclMesh2libgeodesicMesh( pcl_mesh, &distance, &surface, inverse_shift_connected_vertices, shifts );

    /* set time for heat flow */
   // std::cout << "hmTriDistanceEstimateTime..." << std::endl << std::flush;        
   hmTriDistanceEstimateTime( &distance );

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
    hmTriDistanceSetBoundaryConditions( &distance, boundaryConditions );

    /* compute distance */
    std::cout << "hmTriDistanceBuild..." << std::endl << std::flush;        
    hmTriDistanceBuild( &distance );
    std::cout << "distance.time: " << distance.time << std::endl << std::flush;

    

   for(size_t kp = 0; kp < coords_keypoints.size(); ++kp) {

        cv::KeyPoint _kp = coords_keypoints[kp];
        //_kp.pt*=f_scale;

        _kp.pt.x = (int)(coords_keypoints[kp].pt.x + 0.5);
        _kp.pt.y = (int)(coords_keypoints[kp].pt.y + 0.5);
        
        //printf("kp %d: (%f, %f)\n", kp, _kp.pt.x, _kp.pt.y);

        _kp.pt*=f_scale;
        
        _kp.pt.x = (int)(_kp.pt.x); _kp.pt.y = (int)(_kp.pt.y);
        
        
        size_t keypoint_index = _kp.pt.x + _kp.pt.y * pcl_mesh.cloud.width;


        size_t shifted_kpt_index = keypoint_index - shifts[keypoint_index];


        //std::cout << "setSources..." << std::endl << std::flush;        
        setSources( &distance, shifted_kpt_index );

        /* udpate distance function */
       // std::cout << "hmTriDistanceUpdate..." << std::endl <<  std::flush;        
        hmTriDistanceUpdate( &distance );

        /* write distances */
       // std::cout << "saving distances..." << std::endl <<  std::flush;        
        std::vector<double> dists(shifts.size(), -1);

        for (size_t v = 0; v < inverse_shift_connected_vertices.size(); ++v) {
            dists[v + inverse_shift_connected_vertices[v] ] = distance.distance.values[v];
        }
        // interpolate distance map

        std::vector<double> hd_dists;
        hd_dists = interpolate_heatflow(dists,f_scale,coords_keypoints[kp].pt);

        dist_heat_flow.push_back(hd_dists);  


    }
}

void compute_heat_flow_matlab(  const pcl::PolygonMesh &mesh,
                        const std::vector<cv::KeyPoint> &coords_keypoints, 
                        std::vector< std::vector<double> > &dist_heat_flow)
{

}

inline int smoothedSum(const cv::Mat& sum, const cv::Point2f &pt)
{
    // Values used by BRIEF
    //static const int PATCH_SIZE = 48;
    //static const int KERNEL_SIZE = 9;

    static const int HALF_KERNEL = KERNEL_SIZE / 2;

    //int img_y = (int)(kpt.pt.y + 0.5) + pt.y;
    //int img_x = (int)(kpt.pt.x + 0.5) + pt.x;

    int img_y = (int)(pt.y + 0.5);
    int img_x = (int)(pt.x + 0.5);
    return   sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
        - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
        - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
        + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
}



void estimatePointAtIsoCurve(const cv::Mat &heat_flow, const cv::KeyPoint &kp, float dir, float isocurve, cv::Point2f &point)
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

    while (curr_isocurve < id_isocurve ) 
    {
        point = point + step*dir_vec;

        int img_y = (int)(point.y + 0.5);
        int img_x = (int)(point.x + 0.5);
        cv::Vec2f p(img_x, img_y);

        //std::cout << point << std::endl;
        if (cv::norm(p-k) > 24){
            //printf("breaking the loop large norm: norm: %f\n", cv::norm(p-k));
            //getchar();
            break;
        }

        if (point.x < 0 || img_x >= 640 || point.y < 0 || img_y >= 480) {
            PCL_ERROR("breaking the loop beyond the limits\n");
            //getchar();
            break;
        }
    
        if (!std::isnan(heat_flow.at<double>(point)) &&
            //std::fpclassify(heat_flow.at<double>(point)) != FP_ZERO) 
            true)//heat_flow.at<double>(point) > 0.02)
            {
                curr_distance = heat_flow.at<double>(point);
                last_valid_dist = curr_distance; // Save the last valid point
                last_valid_pt.x = point.x; 
                last_valid_pt.y = point.y;
            }
            
        curr_isocurve = (int)(curr_distance / max_isocurve_size);
    }
    
    
    if(!(point.x == last_valid_pt.x && point.y == last_valid_pt.y) && curr_distance - last_valid_dist !=0) // if they differ, we want to estimate a middle point in the hole 
     //that will best explain the position in the RGB image
    {
        cv::Point2f middle_point = point - last_valid_pt;
        double ratio = ((id_isocurve*max_isocurve_size) - last_valid_dist)/(curr_distance - last_valid_dist);
        middle_point = last_valid_pt + (middle_point * ratio );
        
        point.x = middle_point.x;
        point.y = middle_point.y;

            if (point.x < 0 || point.x >= 640 || point.y < 0 || point.y >= 480) 
            {
            PCL_ERROR("breaking the loop beyond the limits\n");
            //getchar();
                
            }
        
    }
    
}

// it seems we have a bug here, all point points on the test pairs do not get the right isocurve
void findPointAtIsoCurve2(const cv::Mat &heat_flow, const cv::KeyPoint &kp, float dir, float isocurve, cv::Point2f &point)
{
    cv::Point2f dir_vec = rotate2d(cv::Point2f(1, 0), dir);
    float step = 0.05;

    static int counter = 0;

    //std::cout << "dir: " << dir_vec << std::endl << std::flush;

    int id_isocurve = (int)(isocurve);

    int curr_isocurve = 0;

    double curr_distance = 0;

    cv::Vec2f k(kp.pt.x, kp.pt.y);

    // take a iso curve equal to the iso curve used in the sampling or the next one (when the
    // point is on a hole).
    point = kp.pt;

    while (curr_isocurve < id_isocurve ) {
        point = point + step*dir_vec;

        int img_y = (int)(point.y + 0.5);
        int img_x = (int)(point.x + 0.5);
        cv::Vec2f p(img_x, img_y);

        //std::cout << point << std::endl;
        if (cv::norm(p-k) > 24){
            //printf("breaking the loop large norm: norm: %f\n", cv::norm(p-k));
            //getchar();
            break;
        }

        if (point.x < 0 || img_x >= 640 || point.y < 0 || img_y >= 480) {
            PCL_ERROR("breaking the loop beyond the limits\n");
            //getchar();
            break;
        }
    
        if (!std::isnan(heat_flow.at<double>(point)) &&
            true) {//std::fpclassify(heat_flow.at<double>(point)) != FP_ZERO) {
            curr_distance = heat_flow.at<double>(point);
        }
        curr_isocurve = (int)(curr_distance / max_isocurve_size);
    }

    //std::cout << std::endl << "---> " << point << std::endl << std::flush;
    if (curr_isocurve - id_isocurve > 0) {
        int img_y = (int)(point.y + 0.5);
        int img_x = (int)(point.x + 0.5);
        cv::Vec2f p(img_x, img_y);
        //printf("Extending the support regions - selected_isocurve: %d, id_isocurve: %d - norm: %f\n", curr_isocurve, id_isocurve, cv::norm(p-k));
    }
}

// it seems we have a bug here, all point points on the test pairs do not get the right isocurve
void findPointAtIsoCurve(const cv::Mat &heat_flow, const cv::KeyPoint &kp, float dir, float isocurve, cv::Point2f &point)
{
    cv::Point2f dir_vec = rotate2d(cv::Point2f(1, 0), dir);
    float step = 0.05;

    static int counter = 0;
    point = kp.pt + step*dir_vec;

    //std::cout << "dir: " << dir_vec << std::endl << std::flush;

    int id_isocurve = (int)(isocurve);

    int curr_isocurve = 0;

    //float accumulated_distance = heat_flow.at<float>(kp.pt); // issue: it is returning "-0"... nan? 
    double curr_distance = 0;


    // take a iso curve equal to the iso curve used in the sampling or the next one (when the
    // point is on a hole).
    cv::Point point_in_the_last_valid_isocurve = kp.pt;
    int last_isocurve_valid = 0;
    while (curr_isocurve < id_isocurve ) {
        point = point + step*dir_vec;
        int img_y = (int)(point.y + 0.5);
        int img_x = (int)(point.x + 0.5);
        if (point.x < 0 || img_x >= 640 || point.y < 0 || img_y >= 480)
            break;

        if (heat_flow.at<double>(point) > 0 && !std::isnan(heat_flow.at<double>(point)) &&
            true){ //std::fpclassify(heat_flow.at<double>(point)) != FP_ZERO) {
            int curr_iso = (int)(curr_distance / max_isocurve_size);
            int new_iso = (int)(heat_flow.at<double>(point) / max_isocurve_size);
            if (abs(curr_iso - id_isocurve) > abs(new_iso - id_isocurve)){
                point_in_the_last_valid_isocurve = point;
                last_isocurve_valid = new_iso;
            }

            curr_distance = heat_flow.at<double>(point);
        }


        curr_isocurve = (int)(curr_distance / max_isocurve_size);
    }

    point = point_in_the_last_valid_isocurve;
    //std::cout << std::endl << "---> " << point << std::endl << std::flush;
    if (last_isocurve_valid - id_isocurve < -3) {
        counter++;
        PCL_ERROR("Shrinking the support regions - selected_isocurve: %d, id_isocurve: %d (%d)\n", last_isocurve_valid, id_isocurve, counter);
        
        //getchar();
    }

    //std::cout << std::endl << std::flush;
}

bool load_keypoints(const std::string &filename, std::vector<cv::KeyPoint> &keypoints)
{
    std::ifstream fin(filename.c_str());

    if (fin.is_open())
    {
        while (!fin.eof()) {
            cv::KeyPoint kp(cv::Point2f(0,0), 6);

            fin >> kp.pt.x;
            fin >> kp.pt.y;

            std::cout << "KeyPoint: " << kp.pt << std::endl << std::flush;
            keypoints.push_back(kp);
        }
        fin.close();
        return true;
    } 
    
    PCL_WARN("Couldn't load file %s\n", filename.c_str());

    return false;
}

bool load_groundtruth_keypoints_csv(const std::string &filename, std::vector<cv::KeyPoint> &keypoints, CSVTable &csv_data)
{
    std::ifstream fin(filename.c_str());
    int id, valid;
    float x,y;
    std::string line;

    if (fin.is_open())
    {
        std::getline(fin, line); //csv header
        while ( std::getline(fin, line) ) 
        {
            if (!line.empty()) 
            {   
                std::stringstream ss;
                char * pch;

                pch = strtok ((char*)line.c_str()," ,");
                while (pch != NULL)
                {
                    ss << std::string(pch) << " ";
                    pch = strtok (NULL, " ,");
                }

                ss >> id >> x >> y >> valid;
                
                if(x<0 || y<0)
                    valid = 0;

                
                std::map<std::string,float> csv_line;
                csv_line["id"] = id;
                csv_line["x"] = x;
                csv_line["y"] = y;
                csv_line["valid"] = valid;
                csv_data.push_back(csv_line);

                //printf("loaded %d %d from file\n",x, y, valid);

                if(valid)
                {
                    cv::KeyPoint kp(cv::Point2f(0,0), 12.0); //6 //7 
                    kp.pt.x = x;
                    kp.pt.y = y;
                    kp.class_id = id;
                    //kp.size = keypoint_scale;
                    kp.octave = 0.0;
                    keypoints.push_back(kp);
                }
            }
        }

        fin.close();
    }
    else
    { 
        PCL_WARN("Unable to open ground truth csv file. Gonna use the detector algorithm.\n"); 
        return false;
    }

    return true;
}

void depth_filtering(const CloudType::Ptr cloud, CloudType::Ptr filtered_cloud)
{
    cv::Mat mask = cv::Mat::zeros( cloud->height, cloud->width, CV_8U );
    cv::Mat depth = cv::Mat::zeros( cloud->height, cloud->width, CV_64FC1 );

    for(int c = 0; c < cloud->width; ++c)
        for(int r = 0; r < cloud->height; ++r) {
            if (std::isnan(cloud->at(c,r).z))
                mask.at<uchar>(r,c) = 255;
            else 
                depth.at<double>(r,c) = cloud->at(c,r).z;
        }

    //cv::imshow("mask depth", mask);
    cv::imwrite("mask_depth.png",mask);
    //cv::waitKey();

    cv::Mat filtered_depth;
    //cv::GaussianBlur( depth, filtered_depth, cv::Size( 15, 15 ), 0, 0 );
    //cv::inpaint(depth, mask, filtered_depth, 5, cv::INPAINT_TELEA);

    cv::Mat _tmp;

    cv::Point minLoc; 
    double minval, maxval;
    minMaxLoc(depth, &minval, &maxval, NULL, NULL);
    printf("%f\n", maxval);
    cv::Mat depthf( cv::Size(640,480), CV_8UC1 );
    depth.convertTo(depthf, CV_8UC1, 255.0/maxval);  //linear interpolation

    //cv::imshow(" depth", depthf);
    cv::imwrite("depth.png",depthf);
    //cv::waitKey();   

    //use a smaller version of the image
    cv::Mat small_depthf; 
    resize(depthf,small_depthf, cv::Size(),0.2,0.2);
    
    cv::Mat small_mask;
    resize(mask,small_mask, cv::Size(),0.2,0.2);

    //inpaint only the "unknown" pixels
    cv::inpaint(small_depthf, small_mask, depth, 5.0, cv::INPAINT_TELEA);
     
    cv::resize(depth, _tmp, depthf.size());
    _tmp.copyTo(filtered_depth);  //add the original signal back over the inpaint

    //bilateralFilter ( src, dst, i, i*2, i/2 );
    pcl::copyPointCloud(*cloud, *filtered_cloud);

    for(int c = 0; c < filtered_depth.cols; ++c)
        for(int r = 0; r < filtered_depth.rows; ++r)
            if (mask.at<uchar>(r,c) == 255)
                filtered_cloud->at(c,r).z = ((float)filtered_depth.at<uchar>(r,c))/255.0 * maxval;

    //cv::imshow("filtered depth", filtered_depth);
    cv::imwrite("filtered_depth.png", filtered_depth);
    //cv::waitKey();
}
// Warning: it changes the number of keypoints (removing keypoints with invalid geodesic distance)
void compute_vector_feature(const CloudType::Ptr cloud, std::string img_path, const std::vector< std::vector<double> > &distance,
                            std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat &descriptors)
{

    cv::Mat img;
    extract_image_from_pointcloud(cloud, img, img_path);

    //const std::string test_pairs_file = sourcedir + "/aux/test_pairs_reaching_holes.txt"; // "test_pairs.txt";
    //const std::string test_pairs_file = sourcedir + "/aux/test_pairs_512.txt";
    const std::string test_pairs_file = sourcedir + "/aux/test_pairs_128.txt";
    std::vector< std::vector<float> > test_pairs;
    
    load_test_pairs(test_pairs_file, test_pairs);

    int num_bytes = test_pairs.size()/8;
    //descriptors = cv::Mat::zeros(keypoints.size(), num_bytes, CV_8U);

    cv::Mat grayImage = img;
    if (img.type() != CV_8U) cvtColor(img, grayImage, CV_BGR2GRAY);

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

        float canonical_angle = compute_canonical_orientation(grayImage, keypoints[kp], umax);

        //cv::Mat heat_flow_colormap;
        //func_display(distance[kp], cloud, heat_flow_colormap);

#if SHOW_TEST_PAIRS

        // shows the keypoint orientation
        std::cout << "Canonical angle: " << canonical_angle * 180.0f/CV_PI << std::endl << std::flush;
        std::cout << "KeyPoint: " << keypoints[kp].pt << std::endl << std::flush;
        std::cout << "Dist: " << img_distance.at<double>(keypoints[kp].pt) << std::endl << std::flush;

        cv::Point2f end_arrow = rotate2d(cv::Point2f(30, 0), canonical_angle);
        cv::arrowedLine(img, keypoints[kp].pt, keypoints[kp].pt + end_arrow, cv::Scalar(255, 0, 0), 1);
        
        
        cv::Point2f end_arrow_dir = rotate2d(cv::Point2f(1, 0), 0);
        cv::arrowedLine(img, keypoints[kp].pt, keypoints[kp].pt + end_arrow_dir, cv::Scalar(0, 0, 255), 1);


        cv::Point2f end_arrow_dir90 = rotate2d(cv::Point2f(1, 0), 3.14159265/2 );
        cv::arrowedLine(img, keypoints[kp].pt, keypoints[kp].pt + end_arrow_dir90, cv::Scalar(0, 255, 0), 1);
        
#endif 

        // Keypoint on a hole (there is no geodesic info for it)
        cv::KeyPoint _keypoint = keypoints[kp];
        _keypoint.pt.x = (int)(keypoints[kp].pt.x + 0.5);
        _keypoint.pt.y = (int)(keypoints[kp].pt.y + 0.5);
        
        if (//img_distance.at<double>(_keypoint.pt) > 0.02 ||
            //std::fpclassify(img_distance.at<double>(_keypoint.pt)) != FP_ZERO ||
            img_distance.at<double>(_keypoint.pt) < 0) {
            printf("Keypoint with invalid distance: %f!\n", img_distance.at<double>(_keypoint.pt));
            std::cout << "Invalid distance: " << _keypoint.pt << "\t\t\t " 
            //<< cloud->at(_keypoint.pt.x, _keypoint.pt.y) 
            << "\t\tDistance: " << img_distance.at<double>(_keypoint.pt) << std::endl << std::flush;

            //show_keypoint_on_heatflow(cloud, distance[kp], keypoints[kp]);
            cv::circle(img, _keypoint.pt, 3, cv::Scalar(0, 0, 0), 1);

            continue;
        }

        valid_keypoints.push_back(_keypoint);

        cv::Mat keypoint_descriptor = cv::Mat::zeros(1, num_bytes, CV_8U);
        uchar* desc = keypoint_descriptor.ptr(0);

        char bitstring[4096] = "";
        uchar feat = 0;

        for (size_t tp = 0; tp < test_pairs.size(); ++tp)
        {
            //std::cout << "=== Test pair # " << tp << "===" << std::endl << std::flush;

            float isocurve = test_pairs[tp][0];
            cv::Point2f pointX;
            float dir = canonical_angle + test_pairs[tp][1];
            estimatePointAtIsoCurve(img_distance, _keypoint, dir, isocurve, pointX);
            //cv::circle(img, pointX, 2, cv::Scalar(0, 0, 255));
            //cv::circle(heat_flow_colormap, pointX, 2, cv::Scalar(0, 0, 0), 1);

            //std::cout << "PointX: " << pointX << std::endl
            //    << "\t" << test_pairs[tp][1] << " " << isocurve << std::endl
            //    << std::flush;

            isocurve = test_pairs[tp][2];
            cv::Point2f pointY;
            dir = canonical_angle + test_pairs[tp][3];
            estimatePointAtIsoCurve(img_distance, _keypoint, dir, isocurve, pointY);

#if SHOW_TEST_PAIRS
            //cv::line(img, pointX, pointY, cv::Scalar(0, 0, 255));
            //cv::imshow("test_pairs", img);
            cv::imwrite("keypoints.png", img);

            printf("Press key to continue...\n");
            //cv::waitKey();
            getchar();
#endif
            uchar vtest = smoothedSum(sum, pointX) < smoothedSum(sum, pointY);
            //bool gtest = abs(img_distance.at<float>(pointX) - img_distance.at<float>(pointY)) > 0.001;

            uchar test = vtest; //| gtest;

            sprintf(bitstring, "%s %d", bitstring, test);
            feat = feat + (test << (tp % 8));
            //std::cout << std::endl << (int)feat;
            if ((tp + 1) % 8 == 0) {
                desc[tp / 8] = feat;
                feat = 0;
                //std::cout << std::endl << bitstring << " (" << (int)desc[tp/8] << ")"<< std::endl;
                //bitstring[0] = '\0';
                //getchar();
            }
        }
#if SHOW_TEST_PAIRS
        //cv::imshow("test pairs and orientation", img);
       // cv::waitKey();
#endif
        //std::cout << descriptors.row(kp) << std::endl << std::flush;
        //std::cout << "\nDescriptor: [" << bitstring << "]" << std::endl << std::flush;

        descriptors.push_back(keypoint_descriptor);

    }
    //cv::imshow("keypoints", img);
    //cv::imwrite("keypoints.png", img);
    //cv::waitKey();

    keypoints.clear();
    keypoints = valid_keypoints;
    printf("Number of valid keypoints: %d\n",valid_keypoints.size());
    printf("Number of valid descriptors: %d\n", descriptors.rows);
}


void filter_keypoints_on_hole(std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Mat>& img_distances)
{
        // Keypoint on a hole (there is no geodesic info for it)
        
    std::vector<cv::KeyPoint> keypoints_;
    std::vector<cv::Mat> img_distances_;
        
    for(size_t kp=0; kp < keypoints.size(); kp++)
    {    
        cv::Mat img_distance = img_distances[kp];
        cv::KeyPoint _keypoint = keypoints[kp];
        _keypoint.pt.x = (int)(keypoints[kp].pt.x + 0.5);
        _keypoint.pt.y = (int)(keypoints[kp].pt.y + 0.5);

        if (std::isnan(img_distance.at<double>(_keypoint.pt)) || //std::fpclassify(img_distance.at<double>(_keypoint.pt)) != FP_ZERO ||
            img_distance.at<double>(_keypoint.pt) < 0) {
            printf("Keypoint with invalid distance: %f!\n", img_distance.at<double>(_keypoint.pt));
            std::cout << "Invalid distance: " << _keypoint.pt /*<< "\t\t\t 3D: " 
            << cloud->at(_keypoint.pt.x, _keypoint.pt.y) 
            << "\t\tDistance: " << img_distance.at<double>(_keypoint.pt) */<< std::endl << std::flush;

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


void compute_features_rotated(const CloudType::Ptr cloud, std::string img_path, const std::vector< std::vector<double> > &distance,
                            std::vector<cv::KeyPoint> &keypoints,
                            std::vector<cv::Mat>& rotated_descriptors)
{
    
    rotated_descriptors.clear();
    std::vector<cv::Mat> mat_distances_vec;

    cv::Mat img;
    extract_image_from_pointcloud(cloud, img, img_path);
    
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
            
            show_keypoint_on_heatflow(img, distance[kp], keypoints[kp]);

            mat_distances_vec.push_back(img_distance);
        }
    
    
    filter_keypoints_on_hole(keypoints,mat_distances_vec);
     
     //Rotated versions of the descriptors (position 0 is the unrotated pattern

    //const std::string test_pairs_file = sourcedir + "aux/test_pairs_reaching_holes.txt"; // "test_pairs.txt";
    const std::string test_pairs_file = sourcedir + "/aux/gaussian_1024.txt";
    std::vector< std::vector<float> > test_pairs;
    
    load_test_pairs(test_pairs_file, test_pairs);

    int num_bytes = test_pairs.size()/8;
    //descriptors = cv::Mat::zeros(keypoints.size(), num_bytes, CV_8U);

    cv::Mat grayImage = img;
    if (img.type() != CV_8U) cvtColor(img, grayImage, CV_BGR2GRAY);

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


        for(int angle=0; angle < nb_angle_bins; angle++)
    {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> valid_keypoints;
        
        printf("Calculating descriptors with rotation = %.2f (%d of %d)\n",angle*(360.f/(double)nb_angle_bins),angle+1,nb_angle_bins);
        
        //cv::namedWindow("test_pairs", cv::WINDOW_NORMAL);
        for (size_t kp = 0; kp < keypoints.size(); ++kp)
        {

            cv::Mat img_distance = mat_distances_vec[kp];

            float canonical_angle = (angle*(360.f/(double)nb_angle_bins)) * (CV_PI/180.f);
            compute_canonical_orientation(grayImage, keypoints[kp], umax);


    #if SHOW_TEST_PAIRS
            // shows the keypoint orientation
            std::cout << "Canonical angle: " << canonical_angle * 180.0f/CV_PI << std::endl << std::flush;
            std::cout << "KeyPoint: " << keypoints[kp].pt << std::endl << std::flush;
            std::cout << "Dist: " << img_distance.at<double>(keypoints[kp].pt) << std::endl << std::flush;

            cv::Point2f end_arrow = rotate2d(cv::Point2f(30, 0), canonical_angle);
            cv::arrowedLine(img, keypoints[kp].pt, keypoints[kp].pt + end_arrow, cv::Scalar(255, 0, 0), 1);
    #endif 

            // Keypoint on a hole (there is no geodesic info for it)
            cv::KeyPoint _keypoint = keypoints[kp];
            _keypoint.pt.x = (int)(keypoints[kp].pt.x + 0.5);
            _keypoint.pt.y = (int)(keypoints[kp].pt.y + 0.5);

            cv::Mat keypoint_descriptor = cv::Mat::zeros(1, num_bytes, CV_8U);
            uchar* desc = keypoint_descriptor.ptr(0);

            char bitstring[4096] = "";
            uchar feat = 0;

            for (size_t tp = 0; tp < test_pairs.size(); ++tp)
            {
                //std::cout << "=== Test pair # " << tp << "===" << std::endl << std::flush;

                float isocurve = test_pairs[tp][0];
                cv::Point2f pointX;
                float dir = canonical_angle + test_pairs[tp][1];
                estimatePointAtIsoCurve(img_distance, _keypoint, dir, isocurve, pointX);

                isocurve = test_pairs[tp][2];
                cv::Point2f pointY;
                dir = canonical_angle + test_pairs[tp][3];
                estimatePointAtIsoCurve(img_distance, _keypoint, dir, isocurve, pointY);

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
                if ((tp + 1) % 8 == 0) {
                    desc[tp / 8] = feat;
                    feat = 0;
                    //std::cout << std::endl << bitstring << " (" << (int)desc[tp/8] << ")"<< std::endl;
                    //bitstring[0] = '\0';
                    //getchar();
                }
            }
    #if SHOW_TEST_PAIRS
            //cv::imshow("test pairs and orientation", img);
           // cv::waitKey();
    #endif
            //std::cout << descriptors.row(kp) << std::endl << std::flush;
            //std::cout << "\nDescriptor: [" << bitstring << "]" << std::endl << std::flush;

            descriptors.push_back(keypoint_descriptor);

        }
    
        
        //cv::imshow("keypoints", img);
        //cv::imwrite("keypoints.png", img);
        //cv::waitKey();

        //keypoints.clear();
        //keypoints = valid_keypoints;
        //printf("Number of valid keypoints: %d\n",valid_keypoints.size());
        //printf("Number of valid descriptors: %d\n", descriptors.rows);
        
        rotated_descriptors.push_back(descriptors);
    
    }
}


void filter_matches(const std::vector<cv::DMatch> &matches, 
                    int threshold, std::vector<cv::DMatch> &filtered_matches)
{

    std::cout<<"Matches before filtering: " << matches.size() << std::endl;
    for (size_t m = 0; m < matches.size(); ++m)
    {
        if (matches[m].distance < threshold)
            filtered_matches.push_back(matches[m]);
    }
    std::cout<<"Matches after filtering: " << filtered_matches.size() << std::endl;


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
            auto start = std::chrono::steady_clock::now();
            dist_mat.at<double>(_i,_j) = cv::norm(desc_query.row(i), desc_tgt.row(j),normType);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end-start;
            matching_sum+= diff.count();

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
 
 
 
std::vector<cv::DMatch> calcAndSaveHammingDistancesNonrigid(std::vector<cv::KeyPoint> kp_query, std::vector<cv::KeyPoint> kp_tgt, 
std::vector<cv::Mat> desc_query, std::vector<cv::Mat> desc_tgt, CSVTable query, CSVTable tgt, std::string file_name)
 {
   //We are going to create a matrix of distances from query to desc and save it to a file 'IMGNAMEREF_IMGNAMETARGET_DESCRIPTORNAME.txt'
   std::vector<cv::DMatch> matches;
   
   std::ofstream oFile(file_name.c_str());
   
   oFile << query.size() << " " << tgt.size() << std::endl;
   
   cv::Mat dist_mat(query.size(),tgt.size(),CV_32S,cv::Scalar(-1));
   
    int c_hits=0;

   for(size_t i=0; i < desc_query[0].rows; i++)
   {
    int menor = 999, menor_idx=-1, menor_i=-1, menor_j = -1;
     
    for(size_t j = 0; j < desc_tgt[0].rows; j++)
      {
        int _i = kp_query[i].class_id; //correct idx
        int _j = kp_tgt[j].class_id; //correct idx
        
        //if(_i < 0 || _i >= dist_mat.rows || _j < 0 || _j >= dist_mat.cols)
        //    std::cout << "Estouro: " << _i << " " << _j << std::endl;
        
        if(!(query[_i]["valid"] == 1 && tgt[_i]["valid"] == 1)) //this match does not exist
          continue;

        if(query[_i]["valid"] == 1 && tgt[_j]["valid"] == 1)
        {
          
          dist_mat.at<int>(_i,_j) = norm_hamming_nonrigid(desc_query, desc_tgt, i, j);
          if(dist_mat.at<int>(_i,_j) < menor )
          {
                        menor = dist_mat.at<int>(_i,_j);
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
      oFile << dist_mat.at<int>(i,j) << " ";
    }
    
    oFile << std::endl;   
    oFile.close(); 
    std::cout <<"Correct matches: " << c_hits << " of " << matches.size() << std::endl;
   
   
   return matches;
 }
 



int norm_hamming_nonrigid(std::vector<cv::Mat>& src, std::vector<cv::Mat>& tgt, int idx_d1, int idx_d2)
{
    std::vector<int> distances;

    auto start = std::chrono::steady_clock::now();
    for(int i=0; i < nb_angle_bins; i++)
        distances.push_back(cv::norm(src[0].row(idx_d1), tgt[i].row(idx_d2),cv::NORM_HAMMING));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start;
    matching_sum+= diff.count();

    size_t min_idx =  std::distance(std::begin(distances),std::min_element(std::begin(distances), std::end(distances)));
        
    return distances[min_idx];
    
}

// Warning: it changes the number of keypoints (removing keypoints with invalid geodesic distance)
void extract_descriptors(   std::string cloud_path,
                            const CloudType::Ptr cloud, 
                            std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat &descriptors)
{
    printf("Extracting descriptors from %d keypoints.\n", keypoints.size());
    CloudType::Ptr filtered_cloud(new CloudType);
    //depth_filtering(cloud, filtered_cloud);

    // create a mesh from the an organized point cloud
    pcl::PolygonMesh mesh;

    pcl::OrganizedFastMesh<PointType> ofm;

    // Set parameters
    ofm.setInputCloud(cloud);
    if (dataset_type == "realdata-standard" || dataset_type == "realdata-smoothed")
        ofm.setMaxEdgeLength(1.0);
    else
        ofm.setMaxEdgeLength(10.0);

    ofm.setTrianglePixelSize(1);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<PointType>::TRIANGLE_ADAPTIVE_CUT);

    // Reconstruct
    std::cout << "Creating mesh... " << std::flush;
    ofm.reconstruct(mesh);
    std::cout << "done\n" << std::flush;

#if 0
    // used to show mesh and point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.3, 0.3, 0.3);

    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
    viewer->addPolygonMesh<PointType>(cloud, mesh.polygons, "mesh", 0);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
    viewer->addCoordinateSystem(0.3);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
#endif 
    // Save mesh in PLY format
    /*
    CloudType cloud_from_mesh;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud_from_mesh);

    std::cout << "#Vertices: " << cloud_from_mesh.size() << "\n" << std::flush;
    std::cout << "#Faces: " << mesh.polygons.size() << "\n" << std::flush;
    std::cout << "done\n" << std::flush;

    std::stringstream filename_output;
    filename_output << outputdir << filename << ".ply";
    std::cout << "saving mesh in " << filename_output.str() << std::endl;
    pcl::io::savePLYFile(filename_output.str(), mesh);
    */


    #ifdef DEBUG_ORIENTATION

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
        
        cv::Mat img, img_tgt;
        extract_image_from_pointcloud(cloud, img, cloud_path + "color.png");
        img_tgt = img.clone();

        cv::Mat grayImage;

        cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(img.cols/2.0,img.rows/2.0),45.0,1.f);

     //cv::namedWindow("test_pairs", cv::WINDOW_NORMAL);
        for (size_t i = 0; i < keypoints.size(); ++i)
        {
             cv::Mat img_buffer = img_tgt.clone();
             int kp=0;
             if (img.type() != CV_8U) cvtColor(img_tgt, grayImage, CV_BGR2GRAY);
             
             float canonical_angle = compute_canonical_orientation(grayImage, keypoints[kp], umax);
             img_tgt = img;
             

            // shows the keypoint orientation
            std::cout << "Canonical angle: " << canonical_angle * 180.0f/CV_PI << std::endl << std::flush;
            std::cout << "KeyPoint: " << keypoints[kp].pt << std::endl << std::flush;

            cv::Point2f end_arrow = rotate2d(cv::Point2f(30, 0), canonical_angle);
            cv::arrowedLine(img_buffer, keypoints[kp].pt, keypoints[kp].pt + end_arrow, cv::Scalar(255, 0, 0), 1);
            
            
            cv::Point2f end_arrow_dir = rotate2d(cv::Point2f(50, 0), 0);
            cv::arrowedLine(img_buffer, keypoints[kp].pt, keypoints[kp].pt + end_arrow_dir, cv::Scalar(0, 0, 255), 1);


            cv::Point2f end_arrow_dir90 = rotate2d(cv::Point2f(50, 0), 3.14159265/2 );
            cv::arrowedLine(img_buffer, keypoints[kp].pt, keypoints[kp].pt + end_arrow_dir90, cv::Scalar(0, 255, 0), 1);


            std::vector<cv::KeyPoint> kp_vec;
            kp_vec.push_back(keypoints[kp]);
            cv::drawKeypoints(img_buffer, kp_vec, img_buffer, cv::Scalar(0,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

            cv::imwrite("keypoints.png", img_buffer);
            printf("Press key to continue...\n");
            getchar();
            
            cv::warpAffine(img_tgt,img_tgt,M, cv::Size());
            
            cv::Point2f rotated = keypoints[kp].pt;
            
            rotated.x-=img.cols/2.0;
            rotated.y-=img.rows/2.0;
            
            rotated = rotate2d(rotated,-45.0*(CV_PI/180.f));
            
            rotated.x+=img.cols/2.0;
            rotated.y+=img.rows/2.0;
            
            keypoints[kp].pt.x = rotated.x;
            keypoints[kp].pt.y = rotated.y;
            
        }

    #endif


    std::vector< std::vector< double > > dist_heat_flow;
    
       if(RECALC_HEATFLOW || !load_heatflow_from_file(cloud_path + ".heatflow",dist_heat_flow))
       {
            printf("Computing heatflow from scratch... it may take a while\n");
            dist_heat_flow.clear();
            compute_heat_flow_c(mesh, keypoints, dist_heat_flow);
            dump_heatflow_to_file(cloud_path + ".heatflow", dist_heat_flow); //save heatflow to disk
       }
       else
            printf("Loaded a precomputed heatflow from file!\n");
    
        

    compute_vector_feature(cloud, cloud_path + "color.png", dist_heat_flow, keypoints, descriptors);

}

// Warning: it changes the number of keypoints (removing keypoints with invalid geodesic distance)
void extract_descriptors_rotated( 
                            std::string cloud_path,
                            const CloudType::Ptr cloud, 
                            std::vector<cv::KeyPoint> &keypoints,
                            std::vector<cv::Mat> &descriptors)
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
    if (dataset_type == "realdata-standard" || dataset_type == "realdata-smoothed")
        ofm.setMaxEdgeLength(1.0); // 1.0
    else
         ofm.setMaxEdgeLength(10.0);

    ofm.setTrianglePixelSize(1);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<PointType>::TRIANGLE_ADAPTIVE_CUT);

    // Reconstruct
    std::cout << "Creating mesh... " << std::flush;
    ofm.reconstruct(mesh);
    std::cout << "done\n" << std::flush;


    std::vector< std::vector< double > > dist_heat_flow;

    if( RECALC_HEATFLOW || !load_heatflow_from_file(cloud_path + ".heatflow",dist_heat_flow))
    {
        dist_heat_flow.clear();
        printf("Computing heatflow from scratch... it may take a while\n");
        compute_heat_flow_c(mesh, keypoints, dist_heat_flow);
        dump_heatflow_to_file(cloud_path + ".heatflow", dist_heat_flow); //save heatflow to disk
    }
    else
        printf("Loaded a precomputed heatflow from file!\n");




    //compute_heat_flow_c(mesh, keypoints, dist_heat_flow);


    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start;
    heatflow_timesum+= diff.count();


    start = std::chrono::steady_clock::now();

    compute_features_rotated(cloud, cloud_path + "-rgb.png", dist_heat_flow, keypoints, descriptors);

    end = std::chrono::steady_clock::now();
    diff = end-start;
    binary_extraction+=diff.count();
    
}

bool load_cloud(const std::string &filename, CloudType::Ptr cloud) 
{
    if (pcl::io::loadPCDFile<PointType>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't load file %s\n", filename.c_str());
        return false;
    }

    std::cout << "Loaded "
        << cloud->width * cloud->height
        << " data points from " << filename << std::endl << std::flush;
        
    //smooth_pcl(cloud, 0.02); //Smooth cloud by using MLS reconstriction

    if (!cloud->isOrganized()) {
        PCL_ERROR("The cloud point is NOT organized\n");
        return false;
    }
    return true;
}

bool sort_matches(cv::DMatch k1, cv::DMatch k2)
{
   return k1.distance < k2.distance;
}

bool sort_kps(cv::KeyPoint k1, cv::KeyPoint k2)
{
    return k1.response > k2.response;
}

void filter_kps_boundingbox(std::vector<cv::KeyPoint>& kps, cv::Point2f pmin, cv::Point2f pmax)
{   //151,4,   546, 462
    std::vector<cv::KeyPoint> new_kps;

    for(int i=0; i< kps.size(); i++)
    {
        if(kps[i].pt.x > pmin.x && kps[i].pt.y > pmin.y && kps[i].pt.x < pmax.x && kps[i].pt.y < pmax.y)
            new_kps.push_back(kps[i]);
    }

    kps = new_kps;

}

void get_best_keypoints(std::vector<cv::KeyPoint>& kps)
{

    filter_kps_boundingbox(kps, cv::Point2f(151,4), cv::Point2f(546,462));

    std::vector<cv::KeyPoint> new_kps;
    std::sort(kps.begin(), kps.end(), sort_kps);
    
    for(int i=0; i < 90 &&  i < kps.size() ; i++)
        new_kps.push_back(kps[i]);
    
    
    kps = new_kps;
}



void save_kps(std::vector<cv::KeyPoint> keypoints, std::string out_filename)
{
        std::ofstream oFile(out_filename.c_str());

        oFile << keypoints.size() << " ";

        for(int i=0; i < keypoints.size(); i++)
        {
            oFile << keypoints[i].pt.x << " " << keypoints[i].pt.y << " ";
        }

        oFile.close();
}

std::vector<cv::DMatch> match_and_filter(std::vector< cv::Mat > d1,  std::vector < cv::Mat > d2, std::vector<cv::KeyPoint> k1, std::vector<cv::KeyPoint> k2, std::string out_filename)
{
    std::vector<cv::DMatch> matches;
    std::vector<int> distances(d2[0].rows);
    std::ofstream oFile(out_filename.c_str());
    cout << "saving in " << out_filename << endl;

    for(size_t i=0; i <d1[0].rows; i++)
    {   
        for(size_t j = 0; j < d2[0].rows; j++)
          distances[j] = norm_hamming_nonrigid(d1, d2, i, j); 

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


int main (int argc, char** argv)
{
    std::string inputdir;
    if (pcl::console::parse_argument(argc, argv, "-inputdir", inputdir) == -1) {
        PCL_ERROR ("Need an input dir! Please use -inputdir to continue.\n");
        return (-1);
    }

    std::string refcloud;
    if (pcl::console::parse_argument(argc, argv, "-refcloud", refcloud) == -1) {
        PCL_ERROR ("Need the refcloud file name! Please use -refcloud to continue.\n");
        return (-1);
    } 

    std::vector<std::string> clouds;
    if (pcl::console::parse_multiple_arguments(argc, argv, "-clouds", clouds) == -1) {
        PCL_ERROR ("Need at leat one other cloud for comparision! Please use -clouds \"cloud1.pcd\" ... \"cloudn.pcd\" to continue.\n");
        return (-1);
    }

    if (pcl::console::parse_argument(argc, argv, "-sourcedir", sourcedir) == -1){
        sourcedir = dirnameOf(__FILE__);
    }

    pcl::console::parse_argument(argc, argv, "-isocurvesize", max_isocurve_size);
    std::cout << "isocurve size: " << max_isocurve_size << std::endl;

    pcl::console::parse_argument(argc, argv, "-kpscale", keypoint_scale);
    std::cout << "keypoint scale: " << keypoint_scale<< std::endl;

    pcl::console::parse_argument(argc, argv, "-pyramidlevels", pyramid_levels);
    std::cout << "Pyramid Levels: " << pyramid_levels<< std::endl;

    std::vector<std::string> descriptor_alg;
    pcl::console::parse_multiple_arguments(argc, argv, "-desc", descriptor_alg);

   std::string keypoint_detector;
    if (pcl::console::parse_argument(argc, argv, "-detector", keypoint_detector) == -1) {
        PCL_ERROR ("Not defined a detector. Using Star detector.\n");
        keypoint_detector = "STAR";
    }

    double dist_threshold;
    if (pcl::console::parse_argument(argc, argv, "-distthreshold", dist_threshold) == -1) {
        PCL_ERROR ("Not defined distance threshold. Using 15.\n");
        dist_threshold = 15;
    }

    if (pcl::console::parse_argument(argc, argv, "-datasettype", dataset_type) == -1) {
        PCL_ERROR ("Did not specified a dataset type. (real or simulated / smoothed or not).\n");
        //keypoint_detector = "STAR";
        return -1;
    }

    /*
    if (pcl::console::parse_argument(argc, argv, "-outputdir", outputdir) == -1)
        outputdir = input_filename.substr(0, pos);    
    */

    if(!strcmp(USE_KEYPOINT_ORIENTATION,""))
        PCL_WARN("[WARNING]: Keypoint orientation normalization disabled for all descriptors.\n");


    std::stringstream out_timings;    

    // Load RGB point cloud
    std::stringstream filename_input;
    filename_input << inputdir << "/" << refcloud;

    
    CloudType::Ptr ref_cloud(new CloudType);
    cv::Mat ref_img;
    cv::Mat ref_img_scaled;
    //if (!load_cloud(filename_input.str(), ref_cloud))
    //    return (-1);
    loadCloudFromPNGImages(inputdir, refcloud, ref_cloud, ref_img);

    // Load keypoint coordinates
    std::vector<cv::KeyPoint> ref_keypoints;
    CSVTable ref_groundtruth;

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
    if (!load_groundtruth_keypoints_csv(keypoints_filename.str(), ref_keypoints,ref_groundtruth)) {
        use_detector = true;
        extract_nonholesmask_from_pointcloud(ref_cloud, nonholesmask);
        feature_detector->detect(ref_img, ref_keypoints);

        //cv::KeyPointsFilter::runByImageBorder(ref_keypoints, cv::Size(640,480), 50);
        get_best_keypoints(ref_keypoints);
        std::cout << "#Detected keypoints [refcloud]: "<< ref_keypoints.size() << std::endl << std::flush;
        save_kps(ref_keypoints, refcloud + ".kp");

    }
    else
    {
        std::cout << "#Loaded keypoints [refcloud]: "<< ref_keypoints.size() << std::endl << std::flush;
    }

 
    #ifdef USE_ROTATED_PATTERNS
        std::vector<cv::Mat> ref_descriptors;
        extract_descriptors_rotated(filename_input.str(), ref_cloud, ref_keypoints, ref_descriptors);
    
    #else
        cv::Mat ref_descriptors;
        extract_descriptors(filename_input.str(), ref_cloud, ref_keypoints, ref_descriptors);
        
    #endif
    

    
    //std::cout << "Refcloud descriptors: " << std::endl << ref_descriptors << std::endl << std::flush;

    for (size_t c = 0; c < clouds.size(); ++c) {
 
        std::stringstream filename_cloud;
        filename_cloud << inputdir << "/" << clouds[c];


        CloudType::Ptr cloud(new CloudType);
        cv::Mat img2;
        cv::Mat img2_scaled;        
        /*if (!load_cloud(filename_cloud.str(), cloud))
        {
            std::cout << "Cannot load " << filename_cloud.str() << std::endl;
            return (-1);
        }
        */
        loadCloudFromPNGImages(inputdir, clouds[c], cloud, img2);

        // Load keypoint coordinates
        std::vector<cv::KeyPoint> keypoints;
        CSVTable groundtruth;


        //extract_image_from_pointcloud(cloud, img2, filename_cloud.str() + "color.png");
        cv::resize(img2, img2_scaled, cv::Size(), keypoint_scale, keypoint_scale);

        std::stringstream keypoints_filename;
        keypoints_filename << inputdir << "/" << clouds[c] << ".csv";
        if (!load_groundtruth_keypoints_csv(keypoints_filename.str(), keypoints, groundtruth)) {
            cv::Mat nonholesmask;
            extract_nonholesmask_from_pointcloud(cloud, nonholesmask);

            feature_detector->detect(img2, keypoints);
            
            //cv::KeyPointsFilter::runByImageBorder(keypoints, cv::Size(640,480), 50);
            
            use_detector = true;
            get_best_keypoints(keypoints);
            std::cout << "#Detected keypoints [cloud]: "<< keypoints.size() << std::endl << std::flush;
            save_kps(keypoints, clouds[c] + ".kp");

            //cv::imshow("mask", nonholesmask);
            //cv::waitKey();
        }
            else
        {
            std::cout << "#Loaded keypoints [dst_clouds]: "<<  keypoints.size() << std::endl << std::flush;
        }


        std::stringstream out_distances;

        #ifdef USE_ROTATED_PATTERNS
            std::vector<cv::Mat> descriptors;
            extract_descriptors_rotated(filename_cloud.str(), cloud, keypoints, descriptors);
            std::cout << "Matches: RefCloud vs " << clouds[c] <<" with sizes " <<ref_descriptors[0].rows << " "<< descriptors[0].rows << std::endl << std::flush;
            out_distances << refcloud <<"__"<< clouds[c] << "__OURS";
            
        #else
            cv::Mat descriptors;
            extract_descriptors(filename_cloud.str(), cloud, keypoints, descriptors);
            std::cout << "Matches: RefCloud vs " << clouds[c] <<" with sizes " <<ref_descriptors.rows << " "<< descriptors.rows << std::endl << std::flush;
            out_distances << refcloud <<"__"<< clouds[c] << "__OURS";
        #endif

        //std::cout << clouds[c] << " descriptors: " << std::endl << descriptors << std::endl << std::flush;

        cv::BFMatcher matcher(cv::NORM_HAMMING, true);

        std::vector<cv::DMatch> o_matches, matches;
        int threshold = dist_threshold;

        cv::Mat outimg; 
        
        if(!use_detector)
        {
            #ifdef USE_ROTATED_PATTERNS
            
                o_matches = calcAndSaveHammingDistancesNonrigid(ref_keypoints,keypoints,ref_descriptors,descriptors,
                ref_groundtruth,groundtruth,out_distances.str()+ ".txt");
                
            #else
            
                o_matches = calcAndSaveDistances(ref_keypoints,keypoints,ref_descriptors,descriptors,
                ref_groundtruth,groundtruth,out_distances.str()+ ".txt", cv::NORM_HAMMING);
                
            #endif
            

            filter_matches(o_matches, threshold, matches);
            cv::drawMatches(ref_img, ref_keypoints, img2, keypoints, matches, outimg);

            std::vector<cv::DMatch> valid_matches = validate_matches(ref_keypoints, keypoints, matches, outimg, ref_img.cols);
                 
            //cv::imshow("ours", outimg);
            std::stringstream out_text;

            out_text << "OURS. # Correct: " << valid_matches.size();
            out_text << " / Acc.: " << std::fixed << std::setprecision(2) << 
            100.0*(valid_matches.size()/(float)matches.size()) << " %" <<
            " / Tot.: " << matches.size();

            cv::putText(outimg, out_text.str().c_str(), cv::Point2f(30,30), cv::FONT_HERSHEY_COMPLEX_SMALL, 
            0.8, cv::Scalar(90,250,90), 1, CV_AA);

            cv::imwrite(out_distances.str()+ ".png",outimg);
            //cv::waitKey();
            
        }
        else
        {
            matches = match_and_filter(ref_descriptors,descriptors, ref_keypoints, keypoints, out_distances.str()+ ".txt"); // match and save .txt with correspondences
            cv::drawMatches(ref_img, ref_keypoints, img2, keypoints, matches, outimg);
            cv::imwrite(out_distances.str()+ ".png",outimg);
        }
        
        
        std::cout <<"------------- Now comparing with #" << descriptor_alg.size() << " baselines -------------" << std::endl;

        out_timings << "Total keypoints - " << "ref: " << ref_keypoints.size() << " target: " << keypoints.size() << std::endl; 
        out_timings << "OURS - " << "heatflow time: " << heatflow_timesum << "s -- binary string: " << binary_extraction << "s "<< "matching: " << matching_sum << std::endl;

        for (size_t  d = 0; d < descriptor_alg.size(); ++d) {
            std::cout << "*** Baseline descriptor *** " <<  descriptor_alg[d] << std::endl;
            binary_extraction = 0;
            matching_sum = 0;
            heatflow_timesum=0;

            cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;

            std::vector<cv::KeyPoint> ref_kps, kps;

            ref_kps = ref_keypoints;
            kps = keypoints;

            for(int i=0; i<ref_kps.size(); i++)
            {
                ref_kps[i].pt.x*=keypoint_scale;
                ref_kps[i].pt.y*=keypoint_scale;
            }

            for(int i=0; i<kps.size(); i++)
            {
                kps[i].pt.x*=keypoint_scale;
                kps[i].pt.y*=keypoint_scale;
            }

            printf("Keypoint Scale = %.2f, #keypoints = %d\n\n",ref_kps[0].size,ref_kps.size());

             cv::Mat descriptors_ref, descriptors_img2;

            if(descriptor_alg[d] == "FREAK") 
            {
                descriptor_extractor = cv::xfeatures2d::FREAK::create();

            }
            else if(descriptor_alg[d] == "BRIEF")
            {
                descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
            } 
            else if(descriptor_alg[d] == "BRISK")
            {
                descriptor_extractor = cv::BRISK::create();
            } 
            else if(descriptor_alg[d] == "ORB") 
            {
                 descriptor_extractor = cv::ORB::create();
            }
            else if(descriptor_alg[d] == "DAISY") 
            {
                 descriptor_extractor =  cv::xfeatures2d::DAISY::create(15.0, 3, 8, 8, cv::xfeatures2d::DAISY::NRM_NONE, cv::noArray(), true, true); 
            }
           
         
 
            std::vector<cv::DMatch> o_matches_competitor, matches_competitor;
            
             if(!use_detector)
            {   auto start = std::chrono::steady_clock::now();        
                descriptor_extractor->compute(ref_img_scaled, ref_kps, descriptors_ref);
                descriptor_extractor->compute(img2_scaled, kps, descriptors_img2);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> diff =  end-start;
                binary_extraction+= diff.count();

                cv::BFMatcher matcher2;

               if(descriptor_alg[d] == "DAISY")
                    matcher2 = cv::BFMatcher(cv::NORM_L1, true);
                else
                    matcher2 = cv::BFMatcher(cv::NORM_HAMMING, true);

            
                std::stringstream out_distances;
                out_distances << refcloud <<"__"<< clouds[c] << "__" << descriptor_alg[d];
                
                
                if(descriptor_alg[d] == "DAISY")
                    o_matches_competitor = calcAndSaveDistances(ref_kps,kps,descriptors_ref,descriptors_img2,
                    ref_groundtruth,groundtruth,out_distances.str()+ ".txt", cv::NORM_L1);
                else
                    o_matches_competitor = calcAndSaveDistances(ref_kps,kps,descriptors_ref,descriptors_img2,
                    ref_groundtruth,groundtruth,out_distances.str()+ ".txt", cv::NORM_HAMMING);               
                    
                out_timings <<  descriptor_alg[d] <<" -- binary string: " << binary_extraction << "s "<< "matching: " << matching_sum << std::endl;


                if(descriptor_alg[d] == "BRISK") //longer descriptor in size
                    filter_matches(o_matches_competitor, threshold*2, matches_competitor);
                else filter_matches(o_matches_competitor, threshold, matches_competitor);

                int correct = 0;
                std::cout << descriptor_alg[d] << " - Matches: RefCloud vs " << clouds[c] << std::endl << std::flush;
    
                cv::drawMatches(ref_img_scaled, ref_kps, img2_scaled, kps, matches_competitor, outimg);


                std::vector<cv::DMatch> valid_matches = validate_matches(ref_kps, kps, matches_competitor, outimg, ref_img_scaled.cols);


                std::stringstream out_text;

                out_text << descriptor_alg[d] << ". # Correct: " << valid_matches.size();
                out_text << " / Acc.: " << std::fixed << std::setprecision(2) << 
                100.0*(valid_matches.size()/(float)matches_competitor.size()) << " %" <<
                " / Tot.: " << matches_competitor.size();

                cv::putText(outimg, out_text.str().c_str(), cv::Point2f(30,30), cv::FONT_HERSHEY_COMPLEX_SMALL, 
                0.8, cv::Scalar(90,250,90), 1, CV_AA);

                cv::imwrite(out_distances.str()+ ".png", outimg);

                //std::cout<<descriptors_ref.row(50)<<std::endl;
                std::cout<<"-------------------------------"<< std::endl;

                matching_sum=0;
                binary_extraction=0;
                
                //cv::waitKey();
                
            }
        
        }
    }
    std::cout <<" --------------------------" << std::endl << out_timings.str();
    return 0;
}

