import numpy as np
import os
import open3d as o3d

dataset = 'KITTI'
seq = '00'
des_method = 'D3F'
total_num = 4541

read_path = "/remote-home/ums_lyxt/slam/fpfh_bow/res_data/"+dataset+"/"+seq+"/D3F_allpoints/"
save_path = "/remote-home/ums_lyxt/slam/fpfh_bow/res_data/"+dataset+"/"+seq+"/D3F_allpoints_pcd/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(total_num):
    pcd = o3d.io.read_point_cloud(read_path+str(i)+".ply")

    o3d.io.write_point_cloud(save_path+str(i)+".pcd", pcd)

    print(i)