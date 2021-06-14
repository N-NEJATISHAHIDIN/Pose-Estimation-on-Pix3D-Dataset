import numpy as np
import math


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


# generate a list of labels for the dataset.[real pose_x, real pose_y, real pose_z, bin of pose_x, bin of pose_y, bin of pose_z]
def generate_label(train_val, n_bins, sett, cat, flag):
    
    #generate the bins margin
    _,bins_degree = np.histogram(1, bins=n_bins, range=(0,360))
    _,bins_degree_2 = np.histogram(1, bins=n_bins, range=(0,360))
    _,bins_radian = np.histogram(1, bins=n_bins, range=(-math.pi,math.pi))
    
    #check if category based or category agnostic
    if (flag == 1): 
        reg_label = np.asarray(train_val_per_category[cat][sett])[:,-3:] 
    else:
        reg_label = np.asarray(train_val[sett])[:,-3:]
        
        
    label = np.zeros((len(reg_label),6))
    
    label[:,:3] = reg_label
    out = np.digitize(reg_label[:,0],bins_degree,right=True)-1
    out1 = np.digitize(reg_label[:,1],bins_degree_2,right=True)-1
    out2 = np.digitize(reg_label[:,2],bins_radian,right=True)-1
    out[out[:]==-1] = 0
    out1[out1[:]==-1] = 0
    out2[out2[:]==-1] = 0
    
       
    #out[out[:]>7] = 0
    #out1[out1[:]>2] = 0
    #out2[out2[:]>3] = 0
    
    label[:,3]=out
    label[:,4]=out1
    label[:,5]=out2
    
    return label