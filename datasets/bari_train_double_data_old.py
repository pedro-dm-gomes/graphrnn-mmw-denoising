import numpy as np
import random
import os

def rotate_translate_jitter_pc(pc, angle, x,y,z):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    for p in range (pc.shape[0]):
                
	    ox, oy, oz = [0,0,0]
	    px, py, pz = pc[p,0:3]
	    
	    # Do via Matrix mutiplication istead
	    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy) + x + (np.random.rand() * (0.05 * 2) - 0.05 )
	    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy) + y + (np.random.rand() * (0.05 * 2) - 0.05 )
	    qz = pz + z  + (np.random.rand() * (0.05 * 2) - 0.05 )
	    pc[p,0:3] = qx, qy, pz
	   
    return pc
    
def shuffle_pc(pc):

  idx = np.arange(len(pc))
  np.random.shuffle(idx)
  pc =pc[idx]
  
  return pc
  
class MMW(object):
    def __init__(
        self,
        root="/home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset",
        seq_length=100,
        num_points=200,
        train=True,
    ):

        self.seq_length = seq_length
        self.num_points = num_points
        #print("num_points:", num_points)
        self.data = []
        
        print("root:", root)
        
        # Load not Rotated dataset 
        root_original_mmW = root + '/Not_Rotated_dataset'
        root_rotated_mmW = root + '/Rotated_dataset'
        # Load rotated dataset
        if ("Not_Rotated" in root):
            ROTATED = False
        else:
            ROTATED = True
            
            
            
        log_nr = 0
        root_original_mmW = root_original_mmW + '/' +str(num_points) + '/train'
        root_rotated_mmW = root_rotated_mmW + '/' +str(num_points) + '/train'
        
        print("root_original_mmW", root_original_mmW)
        if train:
            npy_files = os.listdir(root_original_mmW)
            #npy_files =  ['labels_run_51.npy'] # fast debug
            for run in npy_files:
                file_path_original_mmw = os.path.join(root_original_mmW, run)
                file_path_rotated_mmw = os.path.join(root_rotated_mmW, run)
                
                #print("file_path", file_path)
                npy_run_mmw = np.load(file_path_original_mmw)
                npy_run_mmw = npy_run_mmw[0]

                npy_run_rotated_mmw = np.load(file_path_rotated_mmw)
                npy_run_rotated_mmw = npy_run_rotated_mmw[0]
                
                """  Augmented Dataset """
                #Direction: (It can move backward)
                direction = 1
                if np.random.rand() < 0.25: direction = -1
                if (direction == -1):
                    # Reverse Array order
                    npy_run_mmw = np.flip(npy_run_mmw, axis = 0)
                    npy_run_rotated_mmw = np.flip(npy_run_rotated_mmw, axis = 0)
                
                # Rotate Translate and Jitter the point cloud
                angle =np.random.uniform(-4, 4)
                x  = np.random.uniform(-2, 2)
                y  =np.random.uniform(-2, 2)
                z  =np.random.uniform(-1, 1)

                # For each frame
                for frame in range (0, npy_run_mmw.shape[0] ):
                  npy_run_mmw[frame] = rotate_translate_jitter_pc(npy_run_mmw[frame], angle=0, x=0, y=0, z=0)
                  npy_run_rotated_mmw[frame] = rotate_translate_jitter_pc(npy_run_rotated_mmw[frame], angle=angle, x=x, y=y, z=z)
                  #npy_run[frame] =  shuffle_pc(npy_run[frame])

                npy_run = np.concatenate(( npy_run_mmw,npy_run_rotated_mmw ), axis = 2 )

                self.data.append(npy_run)
            
            print("Train  data", np.shape(self.data) ) #(nr_sequences, )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):


        nr_seq = len(self.data)

        # select random sequnce
        rand = np.random.randint(0, nr_seq)
        log_data = self.data[rand]
        total_lenght = len(log_data)

        direction = 1
        speed = 1
        if direction == 1:
        	start_limit = total_lenght - ( self.seq_length * speed)
        	start = np.random.randint(0, start_limit)
        	end = start + ( self.seq_length * speed)
        cloud_sequence = []
        
        for i in range(start,end, direction * speed):
       
            pc = log_data[i]
            npoints = pc.shape[0]
            cloud_sequence.append(pc)
        points = np.stack(cloud_sequence, axis=0)
       
        return points
