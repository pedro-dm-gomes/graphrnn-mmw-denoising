import os
import numpy as np


def get_dataset_split(split_number):
    
    if (split_number == -1): # For Fast Debug
        test_npy_files =  [3 ]# ,7,49,55,57,68,70,4,8,53,56,62,67,69,6,9,51,58,59,63,66,65]

    if (split_number == 0): # Standart Split
        test_npy_files =  [3,7,49,55,57,68,70]

    if (split_number == 1):
        test_npy_files = [4,8,53,56,62,67,69]

    if (split_number == 2):
        test_npy_files = [6,9,51,58,59,63,66,65]

            
    if (split_number == 7): # Walter current split - split 0 + 64
        test_npy_files =  [3,7,49,55,57,64,68,70]
            

    if (split_number == 11): # Split 11
        test_npy_files =  [ 53,67,74,80,88,96,61,6,64,92,70,85,50,56,57] 
  
        
    if (split_number == 12): # Split 12
        test_npy_files =  [ 98,93,86,75,65,62,4,49,7,51,72] 
                        
    test_npy_files = [ 'labels_run_' + str(num)+ '.npy' for num in test_npy_files]

    return test_npy_files

class MMW(object):
    def __init__(
        self,
        root="/home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset",
        seq_length=100,
        num_points=200,
        train=False,
        split_number =0,               
    ):

        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []

        log_nr = 0
        # print(" MMW DATASET ")
        root = root + '/' +str(num_points)+ '/all_runs_final'
        if not(train):


            npy_files = os.listdir(root)
            test_npy_files = get_dataset_split(split_number)
            npy_files = test_npy_files
    
            # For each run calculate the limit
            for run in npy_files:
                file_path = os.path.join(root, run)
                npy_run = np.load(file_path)
                #print("[LOAD EVAL] 30% File_path", file_path)
                npy_run = npy_run[0]
                
                # Cut 70% of frames
                #d_len = int(npy_run.shape[0]*0.7)
                #npy_run = npy_run[d_len:]
                
                run_size = npy_run.shape[0]
                start = 0
                end = start + seq_length
                while end < run_size:
                    npy_data = npy_run[start:end, :, :]
                    self.data.append(npy_data)
                    start = start + 1 # seq_length
                    end = start + seq_length
            
            print("Test  data", np.shape(self.data) )
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, nr):


        # select a seqeunce
        nr_seq = len(self.data)
        rand = nr
        log_data = self.data[rand]
        total_lenght = len(log_data)

        start_limit = total_lenght - self.seq_length
        start = 0
        cloud_sequence = []

        for i in range(start, start + self.seq_length):
            pc = log_data[i]

            npoints = pc.shape[0]
            cloud_sequence.append(pc)

        points = np.stack(cloud_sequence, axis=0)

        return points
