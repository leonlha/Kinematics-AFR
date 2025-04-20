
import os
import pandas as pd
from multiprocessing import Pool
import geopandas
from shapely.geometry import Polygon

from joblib import Parallel, delayed

def get_coordinates(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    coordinates = []
    for line in lines:
        line = line.strip().split()
        class_id = line[0]
        n = line[-1] if line[-1].isdigit() else "NA"
        xy_coordinates = [(float(line[i]), float(line[i + 1])) for i in range(1, len(line) - 1, 2)]
        if len(xy_coordinates) >= 4:
            coordinates.append([class_id, n, xy_coordinates])

    return coordinates

def process_folder(root_folder, folder_name):
    # Construct the path to the folder containing "labels" subfolder
    folder_path = os.path.join(root_folder, folder_name, 'labels')
    # folder_path = os.path.join(root_folder, folder_name)
    file_names = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".txt")]

    print('file_names:', len(file_names))
    data = []
    for file_name in file_names:
        txt_path = os.path.join(folder_path, file_name)
        coordinates = get_coordinates(txt_path)
        for coord in coordinates:
            class_id, n, polygon_coords = coord
            data.append([folder_name, file_name, class_id, n, polygon_coords])
    return data

def write_to_file(data):

    df = pd.DataFrame(data, columns=["Folder Name", "File Name", "Class ID", "n", "Polygon Coords"])

    centroids = []

    for i in df['Polygon Coords']:
        s = geopandas.GeoSeries(
            [Polygon(i)]
        )
        # centroids.append(s.centroid)

        # Assuming s.centroid.values[0] returns a Shapely Point object
        point = s.centroid.values[0]

        # Access coordinates of the Point
        x_coord = point.x
        y_coord = point.y

        # Append the coordinates as a tuple to the centroids list
        centroids.append((x_coord, y_coord))


    df['centroid'] = centroids

    # Extract frame number from file name
    df["frame_number"] = df["File Name"].str.extract(r"_([\d]+)\.txt", expand=False)

    # Extract file name from the first row of the "File Name" column
    first_file_name = df.loc[0, "File Name"]

    # Extract the part before the first underscore
    video_name = first_file_name.split('_')[0]

    # Drop the specified columns
    df = df.drop(columns=["Folder Name", "File Name", "n", "Polygon Coords"])

    # Sort DataFrame by frame_number (convert to numeric for proper sorting)
    df['frame_number'] = pd.to_numeric(df['frame_number'])
    df = df.sort_values(by='frame_number')

    # Output folder path
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    
    # Create CSV file path based on video_name
    csv_filename = os.path.join(output_folder, f'centroid_{video_name}.csv')
    
    # Write filtered DataFrame to CSV file
    df.to_csv(csv_filename, index=False)

def process_and_write(root_folder, folder_name):
    results = process_folder(root_folder, folder_name)
    write_to_file(results)

if __name__ == "__main__":
    root_folder = r"D:\Research\Phong\ToolTracking_Project\runs\segment"
    folder_names = [name for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name))]
    print(folder_names)
     
    # Specify the number of parallel processes
    num_cores = -1  # Use all available CPU cores
    
    # Execute the tasks in parallel
    Parallel(n_jobs=num_cores)(
        delayed(process_and_write)(root_folder, folder_name)
        for folder_name in folder_names
    )

