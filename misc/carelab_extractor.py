import os
import json
import pickle
import numpy as np
import datetime

def extract_point_clouds_from_json(json_file_path, labels_npy_path, zone_file):
    with open(json_file_path, 'r') as f:
        try:
            data = json.load(f)
        except:
            print(f"Error loading json file {json_file_path}")
            return None
        
    with open(zone_file, 'r') as f:
        try:
            zones = json.load(f)
        except:
            print(f"Error loading zone file {zone_file}")
            return None
    print(f"label file: {labels_npy_path}")
    label_data = np.load(labels_npy_path)
    

    # Map zone names to numbers
    zone_map = {
        "unknown": 0,
        "entrance": 1,
        "right_side_bed": 2,
        "left_side_bed": 3,
        "washroom": 4,
        "sink": 5
    }

    # Build a lookup for zones by (frame_num, timestamp) for fast access
    zone_lookup = {}
    for z in zones:
        key = (z.get("frame_num"), z.get("timestamp"))
        zone_name = z.get("zone", "unknown")
        zone_num = zone_map.get(zone_name, 0)
        zone_lookup[key] = zone_num

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

            # Find matching zone using frame_num and timestamp
            frame_num = frame.get("frameNum") or frame.get("frame_num")
            # Try to match timestamp format in zones file (ISO format)
            iso_timestamp = frame_timestamp.isoformat(timespec='microseconds')

            zone_num = zone_lookup.get((frame_num, iso_timestamp), 0)

            for point in frame_point_clouds:
                point.append(zone_num)

            output.append({"x": np.asarray(frame_point_clouds), "y": label, "timestamp": timestamp_ms, "zone": zone_num})

    return output

def process_scenario_folder(scenario_folder, zone_file, subject, scenario):
    radar_json_path = os.path.join(scenario_folder, 'radar_1', '6843.json')
    labels_npy_path = os.path.join(scenario_folder, 'ts_repetitions.npy')
    if not os.path.exists(radar_json_path):
        print(f"radar_1.json not found in {scenario_folder}")
        return
    
    scenario_data = extract_point_clouds_from_json(radar_json_path, labels_npy_path, zone_file)
    if scenario_data is None:
        return
    
    pickle_file_path = os.path.join("data\\raw_carelab_zoned", f"{subject}_{scenario}.pkl")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(scenario_data, f)
    
    print(f"Saved pickle file for scenario in {scenario_folder}")

def traverse_and_process_folder(root_folder):
    for subject in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]:
        for scenario in range(1, 9):
            scenario_folder = os.path.join(root_folder, subject, str(scenario), "regular_session")
            zone_file = os.path.join(root_folder, "evaluation_results", "radar", f"P{subject}_S{scenario}_radar_positions.json")
            if os.path.isdir(scenario_folder):
                process_scenario_folder(scenario_folder, zone_file, subject, scenario)
            else:
                print(f"Scenario folder {scenario_folder} does not exist")

if __name__ == "__main__":
    root_folder = 'C:\\Users\\Koorosh\\OneDrive - University of Toronto\\Koorosh-CareLab-Data'
    traverse_and_process_folder(root_folder)