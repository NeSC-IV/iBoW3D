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

        // key_feat.resize(keypoint_num, feature_dim);
        // all_feat.resize(pcd_num, feature_dim);

        // read key features
        ifstream fp(key_feature_path+"descriptors_"+to_string(frameID)+".txt");
        string line;
        int rowNum = 0;
        while (getline(fp, line)){ // read each line
            string number;
            istringstream readstr(line); // convert string to stream
            // data in one line are splitted by ","
            for(int j = 0; j < feature_dim; j++){  // correspond to the number of data in each line
                getline(readstr, number, ' '); // get data 
                key_feat.at<float>(rowNum, j) = atof(number.c_str()); // convert string to int 
            }
            rowNum++;
        }

        // read all features
        ifstream fp2(all_feature_path+"descriptors_"+to_string(frameID)+".txt");
        string line2;
        rowNum = 0;
        while (getline(fp2, line2)){ // read each line
            string number;
            istringstream readstr(line2); // convert string to stream
            // data in one line are splitted by ","
            for(int j = 0; j < feature_dim; j++){  // correspond to the number of data in each line
                getline(readstr, number, ' '); // get data 
                all_feat.at<float>(rowNum, j) = atof(number.c_str()); // convert string to int 
            }
            rowNum++;
        }

        // // convert double to float
        // key_feat.convertTo(key_feat, CV_32F);
        // all_feat.convertTo(all_feat, CV_32F);
    }

    void FeatureContainer::update(int pcd_num_, int frameID_)
    {
        pcd_num = pcd_num_;
        frameID = frameID_;

        key_feat.create(keypoint_num, feature_dim, CV_32F);
        all_feat.create(pcd_num, feature_dim, CV_32F);

        key_feat = cv::Mat::zeros(keypoint_num, feature_dim, CV_32F);
        key_feat = cv::Mat::zeros(pcd_num, feature_dim, CV_32F);

        // key_feat.resize(keypoint_num, feature_dim);
        // all_feat.resize(pcd_num, feature_dim);

        // read key features
        ifstream fp(key_feature_path+"descriptors_"+to_string(frameID)+".txt");
        string line;
        int rowNum = 0;
        while (getline(fp, line)){ // read each line
            string number;
            istringstream readstr(line); // convert string to stream
            // data in one line are splitted by ","
            for(int j = 0; j < feature_dim; j++){  // correspond to the number of data in each line
                getline(readstr, number, ' '); // get data 
                key_feat.at<float>(rowNum, j) = atof(number.c_str()); // convert string to int 
            }
            rowNum++;
        }

        // read all features
        ifstream fp2(all_feature_path+"descriptors_"+to_string(frameID)+".txt");
        string line2;
        rowNum = 0;
        while (getline(fp2, line2)){ // read each line
            string number;
            istringstream readstr(line2); // convert string to stream
            // data in one line are splitted by ","
            for(int j = 0; j < feature_dim; j++){  // correspond to the number of data in each line
                getline(readstr, number, ' '); // get data 
                all_feat.at<float>(rowNum, j) = atof(number.c_str()); // convert string to int 
            }
            rowNum++;
        }

        // // convert double to float
        // key_feat.convertTo(key_feat, CV_32F);
        // all_feat.convertTo(all_feat, CV_32F);
    }

}

/* Below codes are for test */

// #include "Data_IO.h"

// int main()
// {
    
//     string dataset = "KITTI";
//     string seq = "00";


//     string dataset_folder;
//     dataset_folder = "/remote-home/ums_lyxt/slam/fpfh_bow/data/"+dataset+"/"+seq+"/velodyne/";

//     int frameID = 0;

//     stringstream lidar_data_path;
//     lidar_data_path << dataset_folder << setfill('0') << setw(6) << frameID << ".bin";
//     cout << lidar_data_path.str() << endl;

//     vector<float> lidar_data = iBoW3D::read_lidar_data_KITTI_CU(lidar_data_path.str());
    
//     // pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>());
//     cout << "number of points: " << lidar_data.size()/4 << endl;

//     // for(std::size_t i = 0; i < lidar_data.size(); i += 4)
//     // {            
//     //     pcl::PointXYZ point;
//     //     point.x = lidar_data[i];
//     //     point.y = lidar_data[i + 1];
//     //     point.z = lidar_data[i + 2];

//     //     current_cloud->push_back(point);
//     // }

//     string key_feature_path = "./descriptor_txt/D3F/"+dataset+"/"+seq+"/";
//     string all_feature_path = "./feature_txt/D3F/"+dataset+"/"+seq+"/";


//     iBoW3D::FeatureContainer test_FC(20, 32, lidar_data.size()/4, frameID, key_feature_path, all_feature_path);
//     cv::Mat text_kf = test_FC.getKeyFeature();
//     // Eigen::MatrixXf text_kf = test_FC.getKeyFeature();
//     cout << text_kf << endl;
//     cout << text_kf.size() << endl;
//     cout << text_kf.rows << endl; 
//     cout << text_kf.cols << endl; 
//     return 0;
// }