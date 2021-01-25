#include "utils.hpp"


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

