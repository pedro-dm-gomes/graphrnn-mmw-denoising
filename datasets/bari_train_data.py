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
        
        if ("Not_Rotated" in root):
            ROTATED = False
        else:
            ROTATED = True
            
        
        
        log_nr = 0
        root = root + '/' +str(num_points) + '/train'
        if train:

            npy_files = os.listdir(root)

            # for fast debug
            #npy_files =  ['labels_run_51.npy']

            for run in npy_files:
                file_path = os.path.join(root, run)
                #print("file_path", file_path)
                npy_run = np.load(file_path)
                npy_run = npy_run[0]

                """  Augmented Dataset """
                #Direction: (It can move backward)
                direction = 1
                if np.random.rand() < 0.25: direction = -1
                if (direction == -1):
                	# Reverse Array order
                	npy_run = np.flip(npy_run, axis = 0)
                
                # Rotate Translate and Jitter the point cloud
                if ROTATED == True:
                    angle =np.random.uniform(-4, 4)
                    x  = np.random.uniform(-2, 2)
                    y  =np.random.uniform(-2, 2)
                else:
                    angle = x = y =0  # Dont rotate and dont translate
                # For each frame
                for frame in range (0, npy_run.shape[0] ):
                  npy_run[frame] = rotate_translate_jitter_pc(npy_run[frame], angle, x, y, 0)
                  #npy_run[frame] =  shuffle_pc(npy_run[frame])
                
                self.data.append(npy_run)
            
            print("Train  data", np.shape(self.data) )
         

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
