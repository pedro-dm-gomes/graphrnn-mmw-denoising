import os
import numpy as np
import random

def rotate_translate_jitter_pc(pc, angle, x,y,z):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    
    for p in range (pc.shape[0]):
                
	    ox, oy, oz = [0,0,0]
	    px, py, pz = pc[p,0:3]
	    
	    # Do via Matrix mutiplication istead

	    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy) + x + np.random.rand() * (0.05 * 2) - 0.05 
	    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy) + y + np.random.rand() * (0.05 * 2) - 0.05 
	    qz = pz + z  +np.random.rand() * (0.05 * 2) - 0.05 
	    pc[p,0:3] = qx, qy, pz
	    
    return pc


def translate_pc(pc, x,y, z):
  
  pc[:,0] = pc[:,0] + x
  pc[:,1] = pc[:,1] + y
  pc[:,2] = pc[:,2] + z
   
  return pc

def rotate_pc(pc, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    
    for p in range (pc.shape[0]):
                
	    ox, oy, oz = [0,0,0]
	    px, py, pz = pc[p,0:3]
	    
	    # Do via Matrix mutiplication istead

	    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
	    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
	    
	    pc[p,0:3] = qx, qy, pz
	    
    return pc

def shuffle_pc(pc):

  idx = np.arange(len(pc))
  np.random.shuffle(idx)
  pc =pc[idx]
  
  return pc
    
def jitter_pc(pc):
  """
  Add random jitter to the pc, jitter is per point
  """ 
  for p in pc:
    #dir_x = random.uniform(-1,1)
    x = np.random.rand() * (0.05 * 2) - 0.05 
    y = np.random.rand() * (0.05 * 2) - 0.05
    z = np.random.rand() * (0.05 * 2) - 0.05
    p[0] = p[0] + x
    p[1] = p[1] + y
    p[2] = p[2] + z
        
    return pc  
  
  return pc

class MMW(object):
    def __init__(
        self,
        root="/home/pedro/Desktop/Datasets/NPYs",
        seq_length=12,
        num_points=4000,
        train=True,
    ):

        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []

        # print("seq_length",seq_length)
        # print("num_points",num_points)

        log_nr = 0
        # print(" MMW DATASET ")

        print("root folder", root)

        if train:
            npy_files = [
                "labels_run_4.npy",
                "labels_run_6.npy",
                "labels_run_8.npy",
                "labels_run_9.npy",
                #"labels_run_10.npy",
                "labels_run_11.npy",
                "labels_run_12.npy"
            ]

            for run in npy_files:
                file_path = os.path.join(root, run)
                npy_run = np.load(file_path)
                #npy_run = npy_run[:, :, :, ]
                print(" LOAD File_path", file_path)
                # print("npy_run", npy_run.shape)
                npy_run = npy_run[0]
                print("run shape", npy_run.shape)
                self.data.append(npy_run)

        else:

            npy_files = [
                "labels_run_7.npy",
                "labels_run_10.npy",
                "labels_run_3.npy"
            ]

            
            # For each run calculate the limit
            for run in npy_files:
            	#print ("run", run)
            	file_path = os.path.join(root, run)
            	npy_run = np.load(file_path)
            	print(" LOAD File_path", file_path)
            	npy_run = npy_run[0]
            	#print("run shape", npy_run.shape)
            	
            	run_size = npy_run.shape[0]
            	start = 0
            	end = start + seq_length
            	while end < run_size:
            		#print("Add sequence")
            		#print("start", start)
            		#print("end", end)
            		
            		npy_data = npy_run[start:end, :, :]
            		#print("npy_data", npy_data.shape)
            		self.data.append(npy_data)
            		
            		
            		start = start + seq_length
            		end = start + seq_length
            
            print("self.data", np.shape(self.data) )
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):

        # print("Get sequence")
        nr_seq = len(self.data)
        # print("nr sequences:", nr_seq)

        # select random sequnce
        rand = np.random.randint(0, nr_seq)
        log_data = self.data[rand]
        total_lenght = len(log_data)

        # Speed of sequence : 0:1
        speed = 1
        #if np.random.rand() < 0.1: speed = 2
        
        #Direction: (It can move backward)
        direction = 1
        if np.random.rand() < 0.25: direction = -1
        direction = -1

        if direction == 1:
        	start_limit = total_lenght - ( self.seq_length * speed)
        	start = np.random.randint(0, start_limit)
        	end = start + ( self.seq_length * speed)
        if direction == -1:
        	start_limit = ( self.seq_length * speed) + 1
        	start = np.random.randint(start_limit, total_lenght )
        	end = start - ( self.seq_length * speed)
        
        #print("[GO] [Seq] %d (of %d)  [%d - %d] (of %d)"% (rand, nr_seq, start, end, total_lenght))
        

        cloud_sequence = []
        cloud_sequence_color = []
        
        # Rotation angle
        angle = np.random.rand()* 2.5 * np.pi
        
        #Translate point cloud
        x = np.random.rand()*4 - 2
        y = np.random.rand()*4 - 2
        

        for i in range(start,end, direction * speed):
            

            pc = log_data[i]
            npoints = pc.shape[0]
            
            # TO DO:: IMPROVE!! THIS IS VERY SLOW!! 
            # Shuffle Point Cloud
            #pc =  shuffle_pc(pc)
            
            # Rotate and Translate the point cloud
            pc = rotate_translate_jitter_pc(pc, angle, x, y, 0)

            
            cloud_sequence.append(pc)



        points = np.stack(cloud_sequence, axis=0)
       
        return points
