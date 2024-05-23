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
    dataset_folder = "/"; // path of point cloud dataset


    // Get GT pose
    string pose_data_path = "/"; // path of gt pose (we use .csv)
    vector<vector<float>> gt_pose = read_ground_truth_pose(pose_data_path);

    float sumPose, GTDist;
    float TDist, deltaDist;

    // Get frame id with loop
    string loopID_data_path = "/"; // path of ID with loop (we use .csv)
    vector<int> frame_loop = read_loop_ID(loopID_data_path);


    int numData = gt_pose.size();


    // parameters of features
    int keypoint_num = 20;
    int feature_dim = 32;
    string key_feature_path = "/"; // path of features of keypoints
    string all_feature_path = "/"; // path of features of all points

    int frameID = 0;
    std::stringstream lidar_data_path;
    lidar_data_path << dataset_folder << frameID << ".ply";
    cout << lidar_data_path.str() << endl;
    
    // int pcd_num;
    std::shared_ptr<geometry::PointCloud> current_pcd(new geometry::PointCloud);

    if(dataset=="KITTI" || dataset=="CU"){
        current_pcd = open3d::io::CreatePointCloudFromFile(lidar_data_path.str());
    }
    else if(dataset=="NCLT"){
        current_pcd = open3d::io::CreatePointCloudFromFile(lidar_data_path.str());
    }

    // parameters of BoW
    int init_pcd_num = 400;
    int init_words_num = 50;
    int words_num_add = 10;
    int update_num = 200;
    double lambda_word = 0.2;
    int near_num = 8;
    int search_num = 5;
    double score_th;
    double fit_th;
    double check_th = 1000000;
    double score_th_2 = 1000000;
    double fit_th_2 = 0;
    bool prior_fit = 1;
    int gap_num = 250;
    int max_iter = 10000;
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

    cout << "Please input fit_th" << endl;
    cin >> fit_th;

    cout << "Please input score_th" << endl;
    cin >> score_th;

    cout << "Please input check_th" << endl;
    cin >> check_th;


    cout << "------------------------------" << endl;
    cout << dataset << endl;
    cout << seq << endl;
    cout << "fit_th: " << fit_th << endl;
    cout << "score_th: " << score_th << endl;
    cout << "check_th: " << check_th << endl;
    cout << "prior_fit: " << prior_fit << endl; 

    // initialize
    iBoW3D::DataBase *pDB = new iBoW3D::DataBase();

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

    
    while (frameID < numData)
    {        
        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << frameID << ".ply";
        cout << lidar_data_path.str() << endl;

        if(dataset=="KITTI" || dataset=="CU"){
            
            std::shared_ptr<geometry::PointCloud> current_pcd(new geometry::PointCloud);

            current_pcd = open3d::io::CreatePointCloudFromFile(lidar_data_path.str());

            piBoW3D->update_current_frame(current_pcd, frameID);

            if (frameID < init_pcd_num-1)
            {
                piBoW3D->update_database();
                frameID++;

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
            
            std::shared_ptr<geometry::PointCloud> current_pcd(new geometry::PointCloud);

            current_pcd = open3d::io::CreatePointCloudFromFile(lidar_data_path.str());

            piBoW3D->update_current_frame(current_pcd, frameID);


            if (frameID < init_pcd_num-1)
            {
                piBoW3D->update_database();
                frameID++;

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


    cout << dataset << endl;
    cout << seq << endl;

    cout << "------------ parameters ----------------" << endl;
    cout << "fit_th: " << fit_th << endl;
    cout << "score_th: " << score_th << endl;
    cout << "check_th: " << check_th << endl;
    cout << "ransac_n: " << ransac_n << endl;
    cout << "prior_fit: " << prior_fit << endl; 

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


    return 0;
}

