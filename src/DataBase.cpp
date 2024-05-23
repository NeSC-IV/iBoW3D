#include "DataBase.h"

using namespace std;

namespace iBoW3D
{
    void DataBase::add_key_feat(cv::Mat new_key_feat)
    {
        key_feat_DB.push_back(new_key_feat);
    }

    void DataBase::add_all_feat(cv::Mat new_all_feat)
    {
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