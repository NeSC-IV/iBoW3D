#include "DataBase.h"

using namespace std;

namespace iBoW3D
{
    void DataBase::add_key_feat(cv::Mat new_key_feat)
    {
        // // convert eigen to mat
        // cv::Mat key_feat_Mat;
        // cv::eigen2cv(new_key_feat, key_feat_Mat);
        // key_feat_Mat.convertTo(key_feat_Mat, CV_32F);

        // key_feat_DB.push_back(key_feat_Mat);

        key_feat_DB.push_back(new_key_feat);
    }

    void DataBase::add_all_feat(cv::Mat new_all_feat)
    {
        // // convert eigen to mat
        // cv::Mat all_feat_Mat;
        // cv::eigen2cv(new_all_feat, all_feat_Mat);
        // all_feat_Mat.convertTo(all_feat_Mat, CV_32F);

        // all_feat_DB.push_back(all_feat_Mat);

        all_feat_DB.push_back(new_all_feat);
    }

    void DataBase::add_pcloud(std::shared_ptr<geometry::PointCloud> pcloud)
    {
        pcloud_DB.push_back(pcloud);
    }

    void DataBase::add_frameIdx(int idx)
    {
        frameIdx.push_back(idx);
    }

    void DataBase::add_DBIdx(int frameID, bool add2DB)
    {
        if(add2DB)
        {
            DBIdx[frameID] = DB_len;
        }
        else
        {
            DBIdx[frameID] = -1;
        }
    }

}