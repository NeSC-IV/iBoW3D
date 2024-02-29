#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <open3d/Open3D.h>

using namespace open3d;
using namespace std;

namespace iBoW3D
{
    class DataBase
    {
        public:
            DataBase():DB_len(0){};

            ~DataBase(){};

            void add_key_feat(cv::Mat new_key_feat);
            void add_all_feat(cv::Mat new_all_feat);
            void add_DB_len(){DB_len++;}
            void add_pcloud(std::shared_ptr<geometry::PointCloud> pcloud);
            void add_frameIdx(int idx);
            void add_DBIdx(int frameID, bool add2DB);

            vector<cv::Mat> get_key_feat(){ return key_feat_DB; }
            vector<cv::Mat> get_all_feat(){ return all_feat_DB; }           
            int get_DB_len(){ return DB_len; }
            std::shared_ptr<geometry::PointCloud> get_cloud_by_idx(int idx){ return pcloud_DB[idx]; }
            cv::Mat get_all_feat_by_idx(int idx){ return all_feat_DB[idx]; }
            vector<int> get_frameIdx(){ return frameIdx; }
            int get_DBIdx_by_idx(int idx){ return DBIdx[idx]; }



        private:
            // the length of two vectors
            int DB_len;

            /* each element in the vector is a matrix of a series features of one pcd,
               for key features, the size is keypointNum * featureNum
               for all features, the size is pointNum * featureNum */
            vector<cv::Mat> key_feat_DB;
            vector<cv::Mat> all_feat_DB;

            vector<std::shared_ptr<geometry::PointCloud>> pcloud_DB;

            vector<int> frameIdx;
            // vector<int> DBIdx;
            map<int, int> DBIdx;

    };

}

