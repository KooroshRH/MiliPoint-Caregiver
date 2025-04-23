import os
import json
import pickle
import numpy as np
import datetime

def extract_point_clouds_from_json(json_file_path, labels_npy_path):
    with open(json_file_path, 'r') as f:
        try:
            data = json.load(f)
        except:
            print(f"Error loading json file {json_file_path}")
            return None

    label_data = np.load(labels_npy_path)
    
    output = []
    for frame in data:
        if 'pointCloud' in frame:
            frame_point_clouds = []
            for point_cloud in frame['pointCloud']:
                frame_point_clouds.append(point_cloud[:3])

            frame_timestamp = datetime.datetime.strptime(frame['timeStamp'], "%d-%m-%Y-%H-%M-%S-%f")
            timestamp_ms = int(frame_timestamp.timestamp() * 1000)
            label = -1
            for i in range(len(label_data) - 1):
                start_label, start_time = label_data[i]
                stop_label, stop_time = label_data[i + 1]
                start_time = datetime.datetime.strptime(start_time, "%d-%m-%Y-%H-%M-%S-%f")
                stop_time = datetime.datetime.strptime(stop_time, "%d-%m-%Y-%H-%M-%S-%f")
                
                if start_label.startswith("start-") and stop_label.startswith("stop-") and start_time <= frame_timestamp <= stop_time:
                    label = start_label.split("start-")[1]
                    break
                
                if start_label == "full_session" and stop_label == "full_session" and start_time <= frame_timestamp <= stop_time:
                    label = "walking"
                    break
                
            output.append({"x": np.asarray(frame_point_clouds), "y": label, "timestamp": timestamp_ms})

    return output

def process_scenario_folder(scenario_folder, subject, scenario):
    radar_json_path = os.path.join(scenario_folder, 'radar_1', '6843.json')
    labels_npy_path = os.path.join(scenario_folder, 'ts_repetitions.npy')
    if not os.path.exists(radar_json_path):
        print(f"radar_1.json not found in {scenario_folder}")
        return
    
    scenario_data = extract_point_clouds_from_json(radar_json_path, labels_npy_path)
    if scenario_data is None:
        return
    
    pickle_file_path = os.path.join("data\\raw_carelab_full_timed", f"{subject}_{scenario}.pkl")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(scenario_data, f)
    
    print(f"Saved pickle file for scenario in {scenario_folder}")

def traverse_and_process_folder(root_folder):
    for subject in ["1", "2", "3F", "4", "5F", "6F", "7", "8", "9", "10", "11F", "12F", "13", "14", "15", "16", "17", "18", "19", "20F"]:
        for scenario in range(1, 9):
            scenario_folder = os.path.join(root_folder, subject, str(scenario), "regular_session")
            if os.path.isdir(scenario_folder):
                process_scenario_folder(scenario_folder, subject, scenario)
            else:
                print(f"Scenario folder {scenario_folder} does not exist")

if __name__ == "__main__":
    root_folder = 'C:\\Users\\Koorosh\\OneDrive - University of Toronto\\Koorosh-CareLab-Data'
    traverse_and_process_folder(root_folder)