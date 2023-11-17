import os
import open3d as o3d
import numpy as np



if __name__ == '__main__':
    dataset = 'KITTI'
    seq = '00'
    des_method = 'D3F'
    total_num = 10

    read_path = "/remote-home/ums_lyxt/slam/fpfh_bow/res_data/"+dataset+"/"+seq+"/D3F_allpoints_plyforpcl/"
    save_path = "/remote-home/ums_lyxt/slam/fpfh_bow/res_data/"+dataset+"/"+seq+"/D3F_allpoints_plyforpcl_2/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(total_num):
        print("processing: " + str(i))

        pcd = o3d.io.read_point_cloud(read_path+str(i)+".ply")
        point = np.array(pcd.points)
        point = point.astype(np.float32) 
        pcd.points = o3d.utility.Vector3dVector(point)
        o3d.io.write_point_cloud(save_path+str(i)+".ply", pcd, write_ascii=False, compressed=True)




