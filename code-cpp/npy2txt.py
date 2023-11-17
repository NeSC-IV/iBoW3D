import numpy as np
import os

dataset = 'KITTI'
seq = '00'
des_method = 'D3F'
total_num = 4541

if not os.path.exists('./descriptor_txt/'+des_method+'/'+dataset+'/'+seq):
    os.makedirs('./descriptor_txt/'+des_method+'/'+dataset+'/'+seq)

if not os.path.exists('./feature_txt/'+des_method+'/'+dataset+'/'+seq):
    os.makedirs('./feature_txt/'+des_method+'/'+dataset+'/'+seq)

for i in range(total_num):
    descriptors = np.load('./descriptor/'+des_method+'/'+dataset+'/'+seq+'/descriptors_'+str(i)+'.npy')

    np.savetxt('./descriptor_txt/'+des_method+'/'+dataset+'/'+seq+'/descriptors_'+str(i)+'.txt', descriptors, fmt='%0.18f')

    descriptors = np.load('./feature/'+des_method+'/'+dataset+'/'+seq+'/descriptors_'+str(i)+'.npy')

    np.savetxt('./feature_txt/'+des_method+'/'+dataset+'/'+seq+'/descriptors_'+str(i)+'.txt', descriptors, fmt='%0.18f')

    print(i)