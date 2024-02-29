// #include <ros/ros.h> 
#include <string>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#include <open3d/Open3D.h>

#include <eigen3/Eigen/Dense>

#include <sstream>
#include <iomanip>
#include <fstream>

#include "FeatureContainer.h"
#include "iBoW3D.h"
#include "Data_IO.h"
#include "DataBase.h"


#include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

using namespace open3d;
using namespace std;
using namespace iBoW3D;

// parameters for read NCLT data
float scaling = 0.005;
float offset = -100.0;


int main(int argc, char** argv)
{
    // ros::init(argc, argv, "BoW3D");
    // ros::NodeHandle nh;  

    // parameters of dataset
    string dataset;
    string seq;
    cout << "Please input dataset" << endl;
    cin >> dataset;
    cout << "Please input seq" << endl;
    cin >> seq;

    float TP1, TP2, FP, FN;
    TP1 = TP2 = FP = FN = 0;
    vector<pair<int, int>> FP_list;

    vector<pair<int, int>> LoopList;

    string dataset_folder;
    // dataset_folder = "/remote-home/ums_lyxt/slam/fpfh_bow/data/"+dataset+"/"+seq+"/velodyne/"; //The last '/' should be added
    dataset_folder = "/remote-home/ums_lyxt/slam/fpfh_bow/res_data/"+dataset+"/"+seq+"/D3F_allpoints/";


    // Get GT pose
    string pose_data_path = "/remote-home/ums_lyxt/slam/fpfh_bow/data/"+dataset+"/"+seq+"/GTposes.csv";
    vector<vector<float>> gt_pose = read_ground_truth_pose(pose_data_path);

    float sumPose, GTDist;
    float TDist, deltaDist;

    // Get frame id with loop
    string loopID_data_path = "/remote-home/ums_lyxt/slam/fpfh_bow/data/"+dataset+"/"+seq+"/loop_lst.csv";
    vector<int> frame_loop = read_loop_ID(loopID_data_path);

    // ros::Rate LiDAR_rate(10); //LiDAR frequency 10Hz

    int numData = gt_pose.size();
    cout << "The number of sequence is: " << numData << endl;


    // parameters of features
    int keypoint_num = 20;
    int feature_dim = 32;
    string key_feature_path = "/remote-home/ums_lyxt/slam/fpfh_bow/descriptor_txt/D3F/"+dataset+"/"+seq+"/";
    string all_feature_path = "/remote-home/ums_lyxt/slam/fpfh_bow/feature_txt/D3F/"+dataset+"/"+seq+"/";

    int frameID = 0;
    std::stringstream lidar_data_path;
    // lidar_data_path << dataset_folder << std::setfill('0') << std::setw(6) << frameID << ".bin";
    lidar_data_path << dataset_folder << frameID << ".ply";
    cout << lidar_data_path.str() << endl;
    
    // int pcd_num;
    std::shared_ptr<geometry::PointCloud> current_pcd(new geometry::PointCloud);

    if(dataset=="KITTI" || dataset=="CU"){

        // vector<float> lidar_data = iBoW3D::read_lidar_data_KITTI_CU(lidar_data_path.str()); 
        // for(std::size_t i = 0; i < lidar_data.size(); i += 4)
        // {            
        //     pcl::PointXYZ point;
        //     point.x = lidar_data[i];
        //     point.y = lidar_data[i + 1];
        //     point.z = lidar_data[i + 2];
    
        //     current_pcd->push_back(point);
        // }
        // pcd_num = lidar_data.size()/4;

        struct timeval t1, t2;
        double timeuse;
        gettimeofday(&t1, NULL);

        current_pcd = open3d::io::CreatePointCloudFromFile(lidar_data_path.str());
        
        gettimeofday(&t2, NULL);
        timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        cout << "time: " << timeuse << endl;

    }
    else if(dataset=="NCLT"){
        // vector<short> lidar_data = iBoW3D::read_lidar_data_NCLT(lidar_data_path.str());
        // for(std::size_t i = 0; i < lidar_data.size(); i += 4)
        // {            
        //     pcl::PointXYZ point;
        //     point.x = lidar_data[i] * scaling + offset;
        //     point.y = lidar_data[i + 1] * scaling + offset;
        //     point.z = lidar_data[i + 2] * scaling + offset;
    
        //     current_pcd->push_back(point);
        // }
        // pcd_num = lidar_data.size()/4;

        struct timeval t1, t2;
        double timeuse;
        gettimeofday(&t1, NULL);

        current_pcd = open3d::io::CreatePointCloudFromFile(lidar_data_path.str());
        
        gettimeofday(&t2, NULL);
        timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        cout << "time: " << timeuse << endl;
    }

    // parameters of BoW
    int init_pcd_num = 400;
    int init_words_num = 50;
    int words_num_add = 10;
    int update_num = 200;
    double lambda_word = 0.2;
    int near_num = 8;
    int search_num;
    double score_th;
    double fit_th;
    double check_th = 1000000;
    double score_th_2;
    double fit_th_2;
    bool prior_fit = 1;
    int gap_num = 250;
    int max_iter;
    int ransac_n;

    bool remove_outliers = false;
    
    if(dataset=="CU")
    {
        ransac_n = 3;
    }
    else
    {
        ransac_n = 4;
    }
    // ransac_n = 4;

    cout << "Please input fit_th" << endl;
    cin >> fit_th;

    cout << "Please input score_th" << endl;
    cin >> score_th;

    cout << "Please input fit_th_2" << endl;
    cin >> fit_th_2;

    cout << "Please input score_th_2" << endl;
    cin >> score_th_2;

    cout << "Please input search_num" << endl;
    cin >> search_num;

    cout << "Please input max_iter" << endl;
    cin >> max_iter;

    cout << "Please input check_th" << endl;
    cin >> check_th;

    // fit_th = 0.985;
    // score_th = 0.8;
    // fit_th_2 = 0.94;
    // score_th_2 = 1.25;
    // max_iter = 10000;


    cout << "------------------------------" << endl;
    cout << dataset << endl;
    cout << seq << endl;
    cout << "fit_th: " << fit_th << endl;
    cout << "score_th: " << score_th << endl;
    cout << "fit_th_2: " << fit_th_2 << endl;
    cout << "score_th_2: " << score_th_2 << endl;
    cout << "max_iter: " << max_iter << endl;
    cout << "check_th: " << check_th << endl;
    cout << "prior_fit: " << prior_fit << endl; 

    // initialize
    iBoW3D::DataBase *pDB = new iBoW3D::DataBase();

    // iBoW3D::FeatureContainer *pFeatureContainer = new iBoW3D::FeatureContainer(keypoint_num, feature_dim, pcd_num, frameID, 
    //                          key_feature_path, all_feature_path);

    iBoW3D::iBoW3D *piBoW3D = new iBoW3D::iBoW3D(current_pcd, frameID, init_words_num, words_num_add, 
                                                 keypoint_num, feature_dim, key_feature_path, all_feature_path,
                                                 pDB, lambda_word, near_num, search_num, 
                                                 score_th, fit_th, check_th, score_th_2, fit_th_2, 
                                                 prior_fit, gap_num,
                                                 max_iter, ransac_n, remove_outliers);

    piBoW3D->update_database();
    
    frameID++;

    int added_num = 0;
    int LoopID = -1;
    Eigen::Matrix4d_u Trans;


    struct timeval t1, t2, t3, t4;
    double timeuse;
    double retrieve_time_sum = 0.0;
    int retrieve_time_cnt = 0;
    
    while (frameID < numData)
    {
        gettimeofday(&t1, NULL);
        
        std::stringstream lidar_data_path;
        // lidar_data_path << dataset_folder << std::setfill('0') << std::setw(6) << frameID << ".bin";
        lidar_data_path << dataset_folder << frameID << ".ply";
        cout << lidar_data_path.str() << endl;

        gettimeofday(&t2, NULL);
        timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        cout << "time1: " << timeuse << endl;

        if(dataset=="KITTI" || dataset=="CU"){
            // vector<float> lidar_data = iBoW3D::read_lidar_data_KITTI_CU(lidar_data_path.str());
            
            std::shared_ptr<geometry::PointCloud> current_pcd(new geometry::PointCloud);
            // pcd_num = lidar_data.size()/4;
            // cout << "number of points: " << pcd_num << endl;

            // for(std::size_t i = 0; i < lidar_data.size(); i += 4)
            // {            
            //     pcl::PointXYZ point;
            //     point.x = lidar_data[i];
            //     point.y = lidar_data[i + 1];
            //     point.z = lidar_data[i + 2];
        
            //     current_pcd->push_back(point);
            // }


            current_pcd = open3d::io::CreatePointCloudFromFile(lidar_data_path.str());
            

            // // downsample
            // float leafSize = 0.3f;
            // pcl::VoxelGrid<pcl::PCLPointCloud2> voxel;
            // voxel.setInputCloud(current_pcd);
            // voxel.setLeafSize(leafSize, leafSize, leafSize);
            // voxel.filter(*current_pcd);

            piBoW3D->update_current_frame(current_pcd, frameID);

            gettimeofday(&t3, NULL);
            timeuse = (t3.tv_sec - t2.tv_sec) + (double)(t3.tv_usec - t2.tv_usec)/1000000.0;
            cout << "time2: " << timeuse << endl;

            if (frameID < init_pcd_num-1)
            {
                piBoW3D->update_database();
                frameID++;

                gettimeofday(&t4, NULL);
                timeuse = (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
                cout << "time3: " << timeuse << endl;

                continue;
            }
            
            else if (frameID == init_pcd_num-1) // database has already been initialized
            {
                piBoW3D->update_database();
                piBoW3D->get_dictionary_and_histogram();
                frameID++;
                continue;
            }

            vector<float> CurrentLocation, LoopLocation;
            CurrentLocation = gt_pose[frameID];

            piBoW3D->get_current_histogram();
            LoopID = piBoW3D->retrieve();

            gettimeofday(&t4, NULL);
            timeuse = (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
            cout << "time3: " << timeuse << endl;
            retrieve_time_sum += timeuse;
            retrieve_time_cnt += 1;

            if (LoopID == -1)
            {
                cout << "No Loop of frame: " << frameID << endl;

                if (find(frame_loop.begin(), frame_loop.end(), frameID) != frame_loop.end())
                {
                    FN++;
                    cout << "False Negative!" << endl;
                }

                added_num++;
                piBoW3D->update_database();
                if (added_num == update_num)
                {
                    piBoW3D->update_dictionary_histograms();
                    added_num = 0;
                }
            }
            else
            {
                cout << "************* Loop! ************" << endl;
                cout << frameID << " Loop with frame: " << LoopID << endl;

                // piBoW3D->pDB->add_DBIdx(0); // update DBIdx
                piBoW3D->update_only_DBIdx(); // update DBIdx

                Trans = piBoW3D->get_trans();

                // LoopLocation & CurrentLocation are all GT poses
                LoopLocation = gt_pose[LoopID];
                sumPose = pow(CurrentLocation[0]-LoopLocation[0], 2) + pow(CurrentLocation[1]-LoopLocation[1], 2);
                GTDist = pow(sumPose, 0.5);
                TDist = pow(pow(Trans(0,3),2)+pow(Trans(1,3),2)+pow(Trans(2,3),2), 0.5);
                deltaDist = abs(GTDist-TDist);
                if (GTDist <= 4){
                    TP1++;
                    // save idx match and add loopId to LoopList
                    LoopList.push_back(make_pair(frameID, LoopID));
                }
                else if(deltaDist <= 4){
                    TP2++;
                    // save idx match and add loopId to LoopList
                    LoopList.push_back(make_pair(frameID, LoopID));
                }
                else{
                    FP++;
                    FP_list.push_back(make_pair(frameID, LoopID));
                }
            }
            
            frameID++;
        }

        else if(dataset=="NCLT"){            
            // vector<float> lidar_data = iBoW3D::read_lidar_data_KITTI_CU(lidar_data_path.str());
            
            std::shared_ptr<geometry::PointCloud> current_pcd(new geometry::PointCloud);
            // pcd_num = lidar_data.size()/4;
            // cout << "number of points: " << pcd_num << endl;

            // for(std::size_t i = 0; i < lidar_data.size(); i += 4)
            // {            
            //     pcl::PointXYZ point;
            //     point.x = lidar_data[i] * scaling + offset;
            //     point.y = lidar_data[i + 1] * scaling + offset;
            //     point.z = lidar_data[i + 2] * scaling + offset;
        
            //     current_pcd->push_back(point);
            // }


            current_pcd = open3d::io::CreatePointCloudFromFile(lidar_data_path.str());
            

            // // downsample
            // float leafSize = 0.3f;
            // pcl::VoxelGrid<pcl::PCLPointCloud2> voxel;
            // voxel.setInputCloud(current_pcd);
            // voxel.setLeafSize(leafSize, leafSize, leafSize);
            // voxel.filter(*current_pcd);

            piBoW3D->update_current_frame(current_pcd, frameID);

            gettimeofday(&t3, NULL);
            timeuse = (t3.tv_sec - t2.tv_sec) + (double)(t3.tv_usec - t2.tv_usec)/1000000.0;
            cout << "time2: " << timeuse << endl;

            if (frameID < init_pcd_num-1)
            {
                piBoW3D->update_database();
                frameID++;

                gettimeofday(&t4, NULL);
                timeuse = (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
                cout << "time3: " << timeuse << endl;

                continue;
            }
            
            else if (frameID == init_pcd_num-1) // database has already been initialized
            {
                piBoW3D->update_database();
                piBoW3D->get_dictionary_and_histogram();
                frameID++;
                continue;
            }

            vector<float> CurrentLocation, LoopLocation;
            CurrentLocation = gt_pose[frameID];

            piBoW3D->get_current_histogram();
            LoopID = piBoW3D->retrieve();

            gettimeofday(&t4, NULL);
            timeuse = (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
            cout << "time3: " << timeuse << endl;
            retrieve_time_sum += timeuse;
            retrieve_time_cnt += 1;

            if (LoopID == -1)
            {
                cout << "No Loop of frame: " << frameID << endl;

                if (find(frame_loop.begin(), frame_loop.end(), frameID) != frame_loop.end())
                {
                    FN++;
                    cout << "False Negative!" << endl;
                }

                added_num++;
                piBoW3D->update_database();
                if (added_num == update_num)
                {
                    piBoW3D->update_dictionary_histograms();
                    added_num = 0;
                }
            }
            else
            {
                cout << "************* Loop! ************" << endl;
                cout << frameID << " Loop with frame: " << LoopID << endl;

                // piBoW3D->pDB->add_DBIdx(0); // update DBIdx
                piBoW3D->update_only_DBIdx(); // update DBIdx

                Trans = piBoW3D->get_trans();

                // LoopLocation & CurrentLocation are all GT poses
                LoopLocation = gt_pose[LoopID];
                sumPose = pow(CurrentLocation[0]-LoopLocation[0], 2) + pow(CurrentLocation[1]-LoopLocation[1], 2);
                GTDist = pow(sumPose, 0.5);
                TDist = pow(pow(Trans(0,3),2)+pow(Trans(1,3),2)+pow(Trans(2,3),2), 0.5);
                deltaDist = abs(GTDist-TDist);
                if (GTDist <= 4){
                    TP1++;
                    // save idx match and add loopId to LoopList
                    LoopList.push_back(make_pair(frameID, LoopID));
                }
                else if(deltaDist <= 4){
                    TP2++;
                    // save idx match and add loopId to LoopList
                    LoopList.push_back(make_pair(frameID, LoopID));
                }
                else{
                    FP++;
                    FP_list.push_back(make_pair(frameID, LoopID));
                }
            }
            
            frameID++;
            
        
        }
    }

    // // save loop identification list result
    // string result_path = "/remote-home/ums_lyxt/slam/fpfh_bow/code_cpp/loop_result/"+dataset+seq;
    // if (access(result_path.c_str(), 0) == -1)  // output_path does not exist
    // {
    //     char command[500] = "mkdir -p ";
    //     strcat(command, result_path.c_str());
    //     system(command);
    // }
    // string output_path_1 = result_path + "/looplist.txt";
    // if (access(output_path_1.c_str(), 0) == -1)  // output_path does not exist
    // {
    //     char command[500] = "touch ";
    //     strcat(command, output_path_1.c_str());
    //     system(command);
    // }
    // iBoW3D::output_loopList(output_path_1, LoopList);

    cout << dataset << endl;
    cout << seq << endl;

    cout << "------------ parameters ----------------" << endl;
    cout << "fit_th: " << fit_th << endl;
    cout << "score_th: " << score_th << endl;
    cout << "fit_th_2: " << fit_th_2 << endl;
    cout << "score_th_2: " << score_th_2 << endl;
    cout << "search_num: " << search_num << endl;
    cout << "max_iter: " << max_iter << endl;
    cout << "check_th: " << check_th << endl;
    cout << "ransac_n: " << ransac_n << endl;
    cout << "prior_fit: " << prior_fit << endl; 

    cout << "------------ retrieve time ----------------" << endl;
    cout << "sum: " << retrieve_time_sum << endl;
    cout << "cnt: " << retrieve_time_cnt << endl;
    cout << "avg: "<< retrieve_time_sum/retrieve_time_cnt << endl;

    int TP = TP1 + TP2;
    cout << "------------ results ----------------" << endl;
    cout << "TP1: " << TP1 << endl;
    cout << "TP2: " << TP2 << endl;
    cout << "TP: " << TP << endl;
    cout << "FP: " << FP << endl;
    cout << "FN: " << FN << endl;

    float P, R, F1;
    P = TP/(TP + FP);
    R = TP/(TP + FN);
    F1 = 2*P*R/(P+R);

    cout << "P: " << P << endl;
    cout << "R: " << R << endl;
    cout << "F1: " << F1 << endl;

    // cout << "FP_list" << endl;
    // for(int i=0; i<(int)(FP_list.size()); i++){
    //     cout << FP_list[i].first << " " << FP_list[i].second << endl;
    // }

    // // save P, R, F1
    // string output_path_2 = result_path + "/results.txt";
    // if (access(output_path_2.c_str(), 0) == -1)  // output_path does not exist
    // {
    //     char command[500] = "touch ";
    //     strcat(command, output_path_2.c_str());
    //     system(command);
    // }
    // ofstream outfile;
    // outfile.open(output_path_2);
    // if (outfile.is_open()) {
    //     outfile << "TP1: " << TP1 << endl;
    //     outfile << "TP2: " << TP2 << endl;
    //     outfile << "TP: " << TP << endl;
    //     outfile << "FP: " << FP << endl;
    //     outfile << "FN: " << FN << endl;

    //     outfile << "P: " << P << endl;
    //     outfile << "R: " << R << endl;
    //     outfile << "F1: " << F1 << endl;

    //     outfile.close();
    // }


    return 0;
}

