#include "utils.hpp"

using namespace H5;

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

void dump_heatflow_to_file(std::string filename, vec2d& heatflow)
{
    FILE * pFile;
    pFile = fopen (filename.c_str(), "wb");
    printf("Saving heatflow in %s \n", filename.c_str());   
    long k, wh;
    double d;
    
    int t;

    k = heatflow.size();
    wh = heatflow[0].size();
    
    fwrite (&k , sizeof(k), 1, pFile);
    fwrite (&wh , sizeof(wh), 1, pFile);

    for(size_t i=0; i < k; i++)
        for(size_t j=0; j < wh; j++)
            fwrite(&(heatflow[0][0]), sizeof(double), 1, pFile);
            
    fclose(pFile);
       
}

template <class T>
class mat3{
    uint32_t NX, NY, NZ;
    public:
        T* data;   
        mat3(int NX, int NY, int NZ): NX(NX), NY(NY), NZ(NZ){data = new T[NX*NY*NZ];}
        ~mat3(){delete data;}
        T& operator()(int i, int j, int k){return data[k + NZ * j + NZ * NY * i]; }
};

int save_hdf5_descs(std::vector<cv::Mat>& descriptors, std::vector<cv::KeyPoint> kps, std::string filename)
{

    const H5std_string  FILE_NAME( filename.c_str() );
    const H5std_string  DATASET_DESCRIPTORS( "descriptors" );
    const H5std_string  DATASET_ID( "id" );
    const int   NX = descriptors[0].rows;                    // dataset dimensions
    const int   NY = descriptors[0].cols;
    const int   NZ = descriptors.size(); // orientation bins
    const int   RANK = 3; //results are stored as a 3D tensor

   /*
    * Data initialization.
    */
    mat3<uint8_t> mat(NX,NY,NZ);
    uint32_t idxs[kps.size()];

    for(int i=0; i < kps.size(); i++)
        idxs[i] = kps[i].class_id;

   for(int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            for(int k=0; k < NZ; k++)
                mat(i,j,k) = descriptors[k].at<uint8_t>(i,j);

      H5File file( FILE_NAME, H5F_ACC_TRUNC );
      /*
       * Define the size of the arrays and create the data space for fixed
       * size dataset.
       */
      // descriptor dataset dimensions
      hsize_t     dimsf[3];              
      dimsf[0] = NX;
      dimsf[1] = NY;
      dimsf[2] = NZ;
      DataSpace dataspace( RANK, dimsf );

    // indices dataset dimensions
      hsize_t dims_id[1];
      dims_id[0] = kps.size();
      DataSpace dataspace_id( 1, dims_id );
      /*
       * Define datatype for the data in the file.
       */
      IntType datatype( PredType::NATIVE_UINT8 );
      datatype.setOrder( H5T_ORDER_LE );

      IntType datatype_id( PredType::NATIVE_UINT32 );
      datatype_id.setOrder( H5T_ORDER_LE );      
      /*
       * Create a new dataset within the file using defined dataspace and
       * datatype and default dataset creation properties.
       */
      DataSet dataset_desc = file.createDataSet( DATASET_DESCRIPTORS, datatype, dataspace );
      DataSet dataset_id = file.createDataSet( DATASET_ID, datatype_id, dataspace_id );
      /*
       * Write the data to the dataset using default memory space, file
       * space, and transfer properties.
       */
      dataset_desc.write( mat.data, PredType::NATIVE_UINT8 );
      dataset_id.write(idxs, PredType::NATIVE_UINT32 );

   return 0;  // successfully terminated

}