#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace std;

namespace iBoW3D
{
    // function to read data from KITTI or Complex Urban
    vector<float> read_lidar_data_KITTI_CU(const std::string lidar_data_path);

    // function to read data from NCLT
    vector<short> read_lidar_data_NCLT(const std::string lidar_data_path);

    // function to read ground truth data
    vector<vector<float>> read_ground_truth_pose(const std::string pose_data_path);

    // function to read loop ID data
    vector<int> read_loop_ID(const std::string loopID_data_path);

    // function to output identification result
    void output_loopList(const std::string output_path, vector<pair<int, int>> loopList);

}

