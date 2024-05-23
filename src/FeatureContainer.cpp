#include "FeatureContainer.h"

using namespace std;

namespace iBoW3D
{
    FeatureContainer::FeatureContainer(int keypoint_num_, int feature_dim_, int pcd_num_, int frameID_, const std::string key_feature_path_, const std::string all_feature_path_):
    keypoint_num(keypoint_num_), feature_dim(feature_dim_), pcd_num(pcd_num_), frameID(frameID_), key_feature_path(key_feature_path_), all_feature_path(all_feature_path_)
    {
        key_feat.create(keypoint_num, feature_dim, CV_32F);
        all_feat.create(pcd_num, feature_dim, CV_32F);

        key_feat = cv::Mat::zeros(keypoint_num, feature_dim, CV_32F);
        all_feat = cv::Mat::zeros(pcd_num, feature_dim, CV_32F);

        // read key features
        ifstream fp(key_feature_path+"descriptors_"+to_string(frameID)+".txt");
        string line;
        int rowNum = 0;
        while (getline(fp, line)){ 
            string number;
            istringstream readstr(line); 
            for(int j = 0; j < feature_dim; j++){  
                getline(readstr, number, ' '); 
                key_feat.at<float>(rowNum, j) = atof(number.c_str()); 
            }
            rowNum++;
        }

        // read all features
        ifstream fp2(all_feature_path+"descriptors_"+to_string(frameID)+".txt");
        string line2;
        rowNum = 0;
        while (getline(fp2, line2)){ 
            string number;
            istringstream readstr(line2); 
            for(int j = 0; j < feature_dim; j++){  
                getline(readstr, number, ' '); 
                all_feat.at<float>(rowNum, j) = atof(number.c_str()); 
            }
            rowNum++;
        }
    }

    void FeatureContainer::update(int pcd_num_, int frameID_)
    {
        pcd_num = pcd_num_;
        frameID = frameID_;

        key_feat.create(keypoint_num, feature_dim, CV_32F);
        all_feat.create(pcd_num, feature_dim, CV_32F);

        key_feat = cv::Mat::zeros(keypoint_num, feature_dim, CV_32F);
        key_feat = cv::Mat::zeros(pcd_num, feature_dim, CV_32F);


        // read key features
        ifstream fp(key_feature_path+"descriptors_"+to_string(frameID)+".txt");
        string line;
        int rowNum = 0;
        while (getline(fp, line)){ 
            string number;
            istringstream readstr(line); 
            for(int j = 0; j < feature_dim; j++){  
                getline(readstr, number, ' '); 
                key_feat.at<float>(rowNum, j) = atof(number.c_str()); 
            }
            rowNum++;
        }

        // read all features
        ifstream fp2(all_feature_path+"descriptors_"+to_string(frameID)+".txt");
        string line2;
        rowNum = 0;
        while (getline(fp2, line2)){ 
            string number;
            istringstream readstr(line2); 
            for(int j = 0; j < feature_dim; j++){  
                getline(readstr, number, ' '); 
                all_feat.at<float>(rowNum, j) = atof(number.c_str()); 
            }
            rowNum++;
        }

    }

}

