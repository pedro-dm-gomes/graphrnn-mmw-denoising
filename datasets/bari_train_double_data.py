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

def get_dataset_split(split_number):
    
    print("split number: ",split_number )
    if (split_number == -1): # For Fast Debug
            test_npy_files =  [3,7,49,55,57,68,70,4,8,53,56,62,67,69,6,9,51,58,59,63,66,65, ]

    if (split_number == 0): # Standart Split
            test_npy_files =  [3,7,49,55,57,68,70]

    if (split_number == 1):
            test_npy_files = [4,8,53,56,62,67,69]

    if (split_number == 2):
            test_npy_files = [6,9,51,58,59,63,66,65]

            
    if (split_number == 7): # Walter current split - split 0 + 64
            test_npy_files =  [3,7,49,55,57,64,68,70]


    if (split_number == 11): # Split 11
        test_npy_files =  [ 9,69,73,3,55, 53,67,74,80,88,96,61,6,64,92,70,85,50,56,57] # test + val
  
        
    if (split_number == 12): # Split 12
        test_npy_files =  [68,77,84,95,87, 98,93,86,75,65,62,4,49,7,51,72] # test + val
                        

    test_npy_files = [ 'labels_run_' + str(num)+ '.npy' for num in test_npy_files]

    return test_npy_files


class MMW(object):
    def __init__(
        self,
        root="/home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset",
        seq_length=100,
        num_points=200,
        train=True,
        split_number =0,
    ):

        self.seq_length = seq_length
        self.num_points = num_points
        #print("num_points:", num_points)
        self.data = []
        
        print("Load data split:", split_number)
        print("root:", root)
        
        # Load not Rotated dataset 
        root_original_mmW = root + '/Not_Rotated_dataset'
        root_rotated_mmW = root + '/Rotated_dataset'

            
            
        log_nr = 0
        root_original_mmW = root_original_mmW + '/' +str(num_points) + '/all_runs_final'
        root_rotated_mmW = root_rotated_mmW + '/' +str(num_points) + '/all_runs_final'
        
        print("root_original_mmW", root_original_mmW)
        if train:
            
            # load all files
            all_npy_files = os.listdir(root_original_mmW)
                        
            # Select the split of dataset
            test_npy_files = get_dataset_split(split_number)
            
            #print("\nTest files", test_npy_files)
            
            # Remove test data from training set
            npy_files = [string for string in all_npy_files if string not in test_npy_files]
            if(split_number== -1): npy_files = ['labels_run_75.npy']
            
            print("\nTrain files", npy_files)
            
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
                #angle =np.random.uniform(-4, 4)
                #x  = np.random.uniform(-2, 2)
                #y  =np.random.uniform(-2, 2)
                #z  =np.random.uniform(-1, 1)

                run_size = npy_run_mmw.shape[0]
                start = 0
                end = start + seq_length
                while end < run_size:
                    
                    npy_run_mmw_data = npy_run_mmw[start:end, :, :]
                    npy_run_rotated_mmw_data = npy_run_rotated_mmw[start:end, :, :]
                    npy_data = np.concatenate( (npy_run_mmw_data,npy_run_rotated_mmw_data ), axis = 2)
                    
                    self.data.append(npy_data)
                    start = start + np.random.randint(1,5) 
                    end = start + seq_length
                    
            
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
            start = 0# np.random.randint(0, start_limit)
            end = start + ( self.seq_length * speed)
        cloud_sequence = []
        
        for i in range(start,end, direction * speed):
       
            pc = log_data[i]
            npoints = pc.shape[0]
            cloud_sequence.append(pc)
        points = np.stack(cloud_sequence, axis=0)
       
        return points
