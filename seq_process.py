import pandas as pd
import numpy as np
import os
import multiprocessing
from tqdm import tqdm
from params import SEQ_LENGTH


def worker(popn_path, folder):
    folder_path = os.path.join(popn_path, folder)
    seq_data = []
    csv_list = [pd.read_csv(os.path.join(folder_path, csv), header=None) for csv in os.listdir(folder_path) if csv.endswith(".csv")]

    # Read csv file
    for csv_pd in csv_list:
        # Get Sequence ID based on the trip purpose change
        from_id = seq_indices(csv_pd.iloc[:,10].tolist())
        
        # Threshold for sequence length
        if len(from_id) >= SEQ_LENGTH:
            tmp_data = csv_pd.iloc[from_id, [3,6,7,9,10,13,4,5]]
            for i in range(len(from_id)- SEQ_LENGTH + 1):
                seq_data.append(tmp_data.iloc[list(range(i, i+SEQ_LENGTH)), :])
            
    return np.array(seq_data)


def seq_indices(lst):
    change_indices = [0] 
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:
            change_indices.append(i)
    return change_indices


# Main
if __name__ == '__main__':
    popn_path = '../Japan_Data/POPN/'
    popn_folder = [folder for folder in os.listdir(popn_path) if folder.startswith("00")]
    csv_list = []
    seq_data = []
    num_cores = multiprocessing.cpu_count()
    
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(worker, [(popn_path, folder) for folder in popn_folder])
            
    '''
    for folder in tqdm(popn_folder):
        folder_path = os.path.join(popn_path, folder)
        csv_list = [pd.read_csv(os.path.join(folder_path, csv), header=None) for csv in os.listdir(folder_path) if csv.endswith(".csv")]

        # Read csv file
        for csv_pd in csv_list:
            # Get Sequence ID based on the trip purpose change
            from_id = seq_indices(csv_pd.iloc[:,10].tolist())
            
            # Threshold for sequence length
            if len(from_id) >= SEQ_LENGTH:
                tmp_data = csv_pd.iloc[from_id, [3,6,7,9,10,13,4,5]]
                for i in range(len(from_id)- SEQ_LENGTH + 1):
                    seq_data.append(tmp_data.iloc[list(range(i, i+SEQ_LENGTH)), :])
            
    '''

    seq_stack = np.concatenate(results, axis=0)
    np.save(f'data/seq_{SEQ_LENGTH}.npy', seq_stack)
    print('Finish storing!')


