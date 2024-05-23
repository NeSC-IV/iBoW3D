#pragma once

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <execution>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
// #include <Eigen/Dense>
// #include <Eigen/Core>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <open3d/Open3D.h>

#include <math.h>

#include "Data_IO.h"
#include "FeatureContainer.h"
#include "DataBase.h"


using namespace open3d;
using namespace std;

namespace iBoW3D
{
    struct result_struct
    {
        double fit = 0.0;
        double score = 100.0;
        Eigen::Matrix4d_u trans = Eigen::Matrix4d_u::Identity();
        int idx = -1;
    };

    struct cand_info
    {
        std::shared_ptr<geometry::PointCloud> pCand;
        std::shared_ptr<pipelines::registration::Feature> cand_features;
        int idx;
    };

    struct reg_info
    {
        std::shared_ptr<geometry::PointCloud> current_pcd;
        std::shared_ptr<pipelines::registration::Feature> current_features;

        std::shared_ptr<geometry::PointCloud> pCand;
        std::shared_ptr<pipelines::registration::Feature> cand_features;
        int idx;
    };


    class iBoW3D
    {
        public:
            iBoW3D(std::shared_ptr<geometry::PointCloud> current_pcd_, 
                   int frameID_, int init_words_num_, int words_num_add_, 
                   const int keypoint_num_, const int feature_dim_, 
                   const std::string key_feature_path_, const std::string all_feature_path_, 
                   DataBase *pDB_,
                   double lambda_word_, int near_num_, int search_num_, 
                   double score_th_, double fit_th_, double chech_th_, double score_th_2_, double fit_th_2_,
                   bool prior_fit_, int gap_num_,
                   int max_iter_, int ransac_n_, bool remove_outliers_);

            ~iBoW3D(){delete pFC;};

            // update current pcd
            void update_current_frame(std::shared_ptr<geometry::PointCloud> current_pcd_, int frameID_);

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
            vector<vector<int>> cand_selection();
            // int geo_verification_icp(vector<int> cand_list);
            int geo_verification_ransac(vector<vector<int>> cand_list);
            void pcd_ransac(pair<int, int> cand_pair, vector<result_struct> & res_list,
                                    std::shared_ptr<geometry::PointCloud> current_pcd,
                                    std::shared_ptr<pipelines::registration::Feature> current_features);
            // void pcd_ransac(pair<int, reg_info> cand_info_pair, vector<result_struct> & res_list);
            void island_ransac(pair<int, vector<int>> island_pair, vector<result_struct> & res_list,
                               std::shared_ptr<geometry::PointCloud> current_pcd,
                               std::shared_ptr<pipelines::registration::Feature> current_features);
            vector<vector<int>> range_check(vector<vector<int>> cand_list);

            // get transformation
            Eigen::Matrix4d_u get_trans(){ return trans; }



        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            
            int frameID;
            int lastLoopID;

            int LoopID;

        private:
            // current frame info
            std::shared_ptr<geometry::PointCloud> current_pcd;
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
            double score_th_2;
            double fit_th_2;
            bool prior_fit;
            int gap_num;
            int max_iter;
            int ransac_n;

            int distance_length;
            
            // remove outliers
            bool remove_outliers;

            // registration result
            Eigen::Matrix4d_u trans;
            

    };
}