#pragma once

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <string>


#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


#include <math.h>

using namespace std;


namespace iBoW3D
{
    class FeatureContainer
    {
        public:
            FeatureContainer(int keypoint_num_, int feature_dim_, int pcd_num_, int frameID_, 
                             const std::string key_feature_path_, const std::string all_feature_path_);

            ~FeatureContainer(){}

            void update(int pcd_num_, int frameID_);

            cv::Mat getKeyFeature(){return key_feat;}
            cv::Mat getAllFeature(){return all_feat;}

            // Eigen::MatrixXf getKeyFeature(){return key_feat;}
            // Eigen::MatrixXf getAllFeature(){return all_feat;}

            // void change_frameID(int new_ID){frameID = new_ID;}

        private:
            int keypoint_num;
            int feature_dim;
            int pcd_num;
            int frameID;
            const std::string key_feature_path;
            const std::string all_feature_path;

            cv::Mat key_feat;
            cv::Mat all_feat;

            // Eigen::MatrixXf key_feat;
            // Eigen::MatrixXf all_feat;

    };
}






