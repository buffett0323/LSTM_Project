'''
UID Trip No Sub-trip No Timestamp Longitude Latitude Gender Age Address Code Work
Trip Purpose Magnification Factor 1 Magnification Factor 2 Transport
The node values shape must be [x, x, x], and store it into npy file format
'''

# %%
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon, MultiPoint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

#%%
# Init settings
Japan_path = "../Japan_Data/POPFLOW"
Japan_path_edit = "../Japan_Data/POPFLOW_EDIT"
Individual_path = "../Japan_Data/INDIVIDUAL"
folder_list = [folder for folder in os.listdir(Japan_path) if folder.startswith("t-csv")]
folder_list_edit = [folder for folder in os.listdir(Japan_path_edit) if folder.startswith("t-csv")]

# Match the folder names
for f in folder_list:
    if f not in folder_list_edit:
        f_path = os.path.join(Japan_path_edit, f)
        os.mkdir(f_path)

PIXEL_SIZE = 1
file_select, ID_all = [], []
max_lng_lis, min_lng_lis, max_lat_lis, min_lat_lis = [], [], [], []

for folder in tqdm(folder_list):
    folderpath = os.path.join(Japan_path, folder)
    file_list = [file for file in os.listdir(folderpath) if file.endswith("00.csv")]
    
    for file in file_list:
        filepath = os.path.join(folderpath, file)
        temp = pd.read_csv(filepath, header=None)
        
        file_select.append(filepath)
        ID_all.append(temp.loc[:,0].tolist())
        max_lng_lis.append(max(temp.loc[:,4].tolist())); min_lng_lis.append(min(temp.loc[:,4].tolist()))
        max_lat_lis.append(max(temp.loc[:,5].tolist())); min_lat_lis.append(min(temp.loc[:,5].tolist()))
        
max_lng, min_lng = math.ceil(max(max_lng_lis)), math.floor(min(min_lng_lis))
max_lat, min_lat = math.ceil(max(max_lat_lis)), math.floor(min(min_lat_lis))
num_rows = int(max_lat * PIXEL_SIZE - min_lat * PIXEL_SIZE) 
num_cols = int(max_lng * PIXEL_SIZE - min_lng * PIXEL_SIZE) 

REGION = Polygon([(min_lng, min_lat), (max_lng, min_lat), (max_lng, max_lat), (min_lng, max_lat)])
SELECT_ID = list(set(ID_all[0]).intersection(*ID_all[1:])) # The data actually starts from 1

print(max_lng, min_lng, max_lat, min_lat)
print(num_rows, num_cols)
print(len(file_select))
print(len(SELECT_ID))

#%%
# Cutting the regions into raster
raster_boundaries = [(i, j) for i in range(num_rows) for j in range(num_cols)]
raster_polygons = [Polygon([(min_lng + x * ((max_lng - min_lng) / num_cols),        min_lat + y * ((max_lat - min_lat) / num_rows)), 
                            (min_lng + (x + 1) * ((max_lng - min_lng) / num_cols),  min_lat + y * ((max_lat - min_lat) / num_rows)), 
                            (min_lng + (x + 1) * ((max_lng - min_lng) / num_cols),  min_lat + (y + 1) * ((max_lat - min_lat) / num_rows)), 
                            (min_lng + x * ((max_lng - min_lng) / num_cols),        min_lat + (y + 1) * ((max_lat - min_lat) / num_rows))]) for x, y in raster_boundaries]

# Algorithm for data processing and clustering
edit_data = []

for file in tqdm(file_select):
    temp = pd.read_csv(file, header=None)  
    lng, lat = temp.loc[:,4].tolist(), temp.loc[:,5].tolist()
    points = [Point(lng[i], lat[i]) for i in range(temp.shape[0])]  
    
    # Points containing in the polygons
    point_contain = [polygon.contains(points).tolist() for polygon in raster_polygons] # length: num_rows * num_cols
    point_counts = [item.count(True) for item in point_contain] # Seemingly unuseful
    
    # Find the true value in the small lists
    point_contain_np = np.array(point_contain)
    point_indices = np.where(point_contain_np)
    
    # Match the polygon back to the folder
    point_match_raster = [999 for i in range(temp.shape[0])]
    for i, _ in enumerate(point_indices[0]):
        point_match_raster[point_indices[1][i]] = point_indices[0][i]
    temp[14] = point_match_raster

    # Temporarily add column names
    temp.columns = ['ID', 'Trip No', 'Sub-trip No', 'Timestamp', 'Longitude', 'Latitude', 'Gender', 'Age', 'Address Code', 
                    'Work', 'Trip Purpose', 'Magnification Factor 1', 'Magnification Factor 2', 'Transport', 'Y']
    filter_data = temp[temp['ID'].isin(SELECT_ID)]
    filter_df = filter_data.drop_duplicates(subset='ID')
    filter_df = filter_df.reset_index(drop=True)
    pd.DataFrame(filter_df).to_csv(file.replace('POPFLOW', 'POPFLOW_EDIT'), index=False)
    edit_data.append(filter_df)


# %%
# Individual's data
for i in tqdm(range(len(SELECT_ID))):
    id_data = [data.loc[i].values.tolist() for data in edit_data]
    sorted_df = pd.DataFrame(id_data).sort_values(3) # Sort based on Time stamp
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df.to_csv(f'{Individual_path}/IND_{i}.csv', index=False)

# %%
