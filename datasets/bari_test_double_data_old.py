import os
import numpy as np


class MMW(object):
    def __init__(
        self,
        root="/home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset",
        seq_length=100,
        num_points=200,
        train=False,
    ):

        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []

        log_nr = 0
        print("root", root)
        
        root_original_mmW = root + '/Not_Rotated_dataset'
        root_rotated_mmW = root + '/Rotated_dataset'
        
        
        root_original_mmW = root_original_mmW + '/' +str(num_points)+ '/test'
        root_rotated_mmW = root_rotated_mmW + '/' +str(num_points)+ '/test'
        print("root_original_mmW", root_original_mmW)
        print("root_rotated_mmW", root_rotated_mmW)
        
        if not(train):

            npy_files = os.listdir(root_original_mmW)
            # For each run calculate the limit
            
            for run in npy_files:
                file_path_original_mmw = os.path.join(root_original_mmW, run)
                file_path_rotated_mmw = os.path.join(root_rotated_mmW, run)
                
                print("file_path_original_mmw", file_path_original_mmw)
                print("file_path_rotated_mmw", file_path_rotated_mmw)
                
                npy_run_mmw = np.load(file_path_original_mmw)
                npy_run_mmw = npy_run_mmw[0]

                npy_run_rotated_mmw = np.load(file_path_rotated_mmw)
                npy_run_rotated_mmw = npy_run_rotated_mmw[0]
            	

                run_size = npy_run_mmw.shape[0]
                start = 0
                end = start + seq_length
                while end < run_size:
                    #print("start", start)
                    #print("end", end)
                    #print("npy_run_mmw.shape")
                    npy_run_mmw_data = npy_run_mmw[start:end, :, :]
                    npy_run_rotated_mmw_data = npy_run_rotated_mmw[start:end, :, :]
                    npy_data = np.concatenate( (npy_run_mmw_data,npy_run_rotated_mmw_data ), axis = 2)
                    #print("npy_data", npy_data.shape)
                    self.data.append(npy_data)
                    start = start + seq_length
                    end = start + seq_length
            
            print("Test data", np.shape(self.data) )
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, nr):


        # select a seqeunce
        nr_seq = len(self.data)
        rand = nr
        log_data = self.data[rand]
        total_lenght = len(log_data)

        # print("nr frames:", total_lenght)
        start_limit = total_lenght - self.seq_length
        start = 0

        #print("[Seq] %d (of %d)  [%d - %d] (of %d)"% (rand, nr_seq, start, start + self.seq_length, total_lenght) )

        cloud_sequence = []
        cloud_sequence_color = []

        for i in range(start, start + self.seq_length):
            pc = log_data[i]

            npoints = pc.shape[0]
            # sample_idx = np.random.choice(npoints, self.num_points, replace=False)

            cloud_sequence.append(pc)

        points = np.stack(cloud_sequence, axis=0)

        return points
