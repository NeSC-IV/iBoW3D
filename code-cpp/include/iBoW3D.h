#pragma once

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <pcl/kdtree/impl/kdtree_flann.hpp> 
#include <pcl/impl/point_types.hpp>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h> 
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/common/common.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
// #include <Eigen/Dense>
// #include <Eigen/Core>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <math.h>

#include "Data_IO.h"
#include "FeatureContainer.h"
#include "DataBase.h"

const int N = 32;
typedef pcl::Histogram<N> FeatureT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

using namespace std;

namespace iBoW3D
{
    struct result_struct
    {
        double fit;
        double score;
        Eigen::Matrix4f trans;
        int idx;
    };


    class iBoW3D
    {
        public:
            iBoW3D(pcl::PointCloud<pcl::PointXYZ>::Ptr current_pcd_, 
                   int frameID_, int init_words_num_, int words_num_add_, 
                   const int keypoint_num_, const int feature_dim_, 
                   const std::string key_feature_path_, const std::string all_feature_path_, 
                   DataBase *pDB_,
                   double lambda_word_, int near_num_, int search_num_, double score_th_, double fit_th_, double chech_th_, 
                   bool prior_fit_, int gap_num_);

            ~iBoW3D(){delete pFC;};

            // update current pcd
            void update_current_frame(pcl::PointCloud<pcl::PointXYZ>::Ptr current_pcd_, int frameID_);

            // // update last loop ID
            // void update_lastLoopID(int newID){lastLoopID = newID;}

            // create BoW dictionary; get and save histograms of each seen scans
            void get_dictionary_and_histogram();

            // update BoW dictionary
            void update_dictionary_histograms();

            // update database (can also initialize)
            void update_database();
            void update_only_DBIdx();
            // void update_database_wo_key_feat();

            // get current histogram
            void get_current_histogram();

            // retrieve the loop ID
            int retrieve();
            vector<int> cand_selection();
            int geo_verification_icp(vector<int> cand_list);
            int geo_verification_ransac(vector<int> cand_list);
            vector<int> range_check(vector<int> cand_list);

            // get transformation
            Eigen::Matrix4f get_trans(){ return trans; }



        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            
            int frameID;
            int lastLoopID;

            int LoopID;

        private:
            // current frame info
            pcl::PointCloud<pcl::PointXYZ>::Ptr current_pcd;
            int pcd_num;

            const std::string key_feature_path;
            const std::string all_feature_path;

            FeatureContainer *pFC;
            Eigen::VectorXd current_hist;

            // database
            DataBase *pDB;
            vector<Eigen::VectorXd> hist_DB;

            // BoW dictionary
            int words_num; // the number of words in the dictionary
            int words_num_add; // the added number of words in the dictionary
            cv::Mat labels; // save class label of each feature
            cv::Mat centers; // save the data of each center of cluster
            // Mat centers(words_num,1,points.type()
            Eigen::ArrayXd idf;


            // parameters about features
            int keypoint_num;
            int feature_dim;

            // hyperparameters
            double lambda_word;
            int near_num;
            int search_num;
            double score_th;
            double fit_th;
            double check_th;
            bool prior_fit;
            int gap_num;
            

            // registration result
            Eigen::Matrix4f trans;

    };
}