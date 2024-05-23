#include "Data_IO.h"

using namespace std;

namespace iBoW3D
{
    // function to read data from KITTI or Complex Urban
    vector<float> read_lidar_data_KITTI_CU(const std::string lidar_data_path)
    {
        std::ifstream lidar_data_file;
        lidar_data_file.open(lidar_data_path, std::ifstream::in | std::ifstream::binary);
        if(!lidar_data_file)
        {
            cout << "Read End..." << endl;
            exit(-1);
        }

        lidar_data_file.seekg(0, std::ios::end);
        const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
        lidar_data_file.seekg(0, std::ios::beg);

        std::vector<float> lidar_data_buffer(num_elements);
        lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
        return lidar_data_buffer;
    }


    // function to read data from NCLT
    vector<short> read_lidar_data_NCLT(const std::string lidar_data_path)
    {
        std::ifstream lidar_data_file;
        lidar_data_file.open(lidar_data_path, std::ifstream::in | std::ifstream::binary);
        if(!lidar_data_file)
        {
            cout << "Read End..." << endl;
            exit(-1);
        }

        lidar_data_file.seekg(0, std::ios::end);
        const size_t num_elements = lidar_data_file.tellg() / sizeof(short);
        lidar_data_file.seekg(0, std::ios::beg);

        std::vector<short> lidar_data_buffer(num_elements);
        lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(short));
        return lidar_data_buffer;
    }


    vector<vector<float>> read_ground_truth_pose(const std::string pose_data_path)
    {
        vector<vector<float>> gt_pose;
        ifstream fp1(pose_data_path); 
        string line1;
        while (getline(fp1,line1)){ 
            vector<float> data_line;
            string number;
            istringstream readstr(line1); 
            for(int j = 0;j < 3;j++){  
                getline(readstr,number,','); 
                data_line.push_back(atof(number.c_str())); 
            }
            gt_pose.push_back(data_line);
        }
        return gt_pose;
    }

    vector<int> read_loop_ID(const std::string loopID_data_path)
    {
        vector<int> frame_loop;
        ifstream fp2(loopID_data_path); 
        string line2;
        while (getline(fp2,line2)){ 
            int frameID;
            string number;
            istringstream readstr(line2); 
            for(int j = 0;j < 1;j++){ 
                getline(readstr,number,','); 
                frameID = atoi(number.c_str());  
            }
            frame_loop.push_back(frameID); 
        }

        return frame_loop;
    }
    
    void output_loopList(const std::string output_path, vector<pair<int, int>> loopList)
    {
        ofstream outfile;
        outfile.open(output_path);
        if (outfile.is_open()) {
            for (auto& p : loopList) {
                outfile << p.first << ", " << p.second << endl;
            }
            outfile.close();
        }
    }
}
