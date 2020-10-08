#include "clouds.hpp"

void loadCloudFromPNGImages(const std::string &inputdir, const std::string &filename, CloudType::Ptr cloud, cv::Mat &rgb, int pyramid_levels)
{

    printf("Loading cloud from %s/%s\n", inputdir.c_str(), filename.c_str());
    cv::FileStorage fs;
    cv::Mat in_depth, depth, matcloud;
    cv::Mat K;
    cv::Mat global_K;
    cv::Mat global_Kinv;

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

void extract_image_from_pointcloud(const CloudType::Ptr cloud, cv::Mat &rgb, std::string img_path, std::string dataset_type)
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