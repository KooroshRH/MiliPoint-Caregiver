import os
import json
import pickle
import numpy as np
import datetime
from scipy.spatial.distance import pdist

# Zone definitions for carelab environment
ZONES = [
    {"label": "entrance", "x_min": 3.5, "x_max": 6.0, "y_min": 0.0, "y_max": 2.0, "zone_id": 1},
    {"label": "washroom", "x_min": 4.0, "x_max": 6.5, "y_min": 2.0, "y_max": 3.0, "zone_id": 4},
    {"label": "left_side_bed", "x_min": 2.5, "x_max": 3.5, "y_min": 1.4, "y_max": 3.0, "zone_id": 3},
    {"label": "right_side_bed", "x_min": 0.0, "x_max": 1.4, "y_min": 1.4, "y_max": 3.0, "zone_id": 2},
    {"label": "sink", "x_min": 0.0, "x_max": 3.5, "y_min": 0.0, "y_max": 1.4, "zone_id": 5},
]

# Radar configuration
RADAR_AZIMUTH = 285  # degrees
RADAR_X = 0.14       # radar position x
RADAR_Y = 0.35       # radar position y

def assign_zone_to_point(x, y):
    """
    Assign zone ID to a point based on its x, y coordinates.

    Args:
        x: X coordinate (after radar transformation)
        y: Y coordinate (after radar transformation)

    Returns:
        zone_id: Integer zone ID (0 if no zone matches)
    """
    for zone in ZONES:
        if (zone["x_min"] <= x <= zone["x_max"] and
            zone["y_min"] <= y <= zone["y_max"]):
            return zone["zone_id"]
    return 0  # Unknown zone

def process_point_with_zone(point):
    """
    Transform a single point and assign its zone based on its individual position.

    Args:
        point: [x, y, z] coordinates (before transformation)

    Returns:
        [x, y, z, zone_id] coordinates (after transformation and individual zone assignment)
    """
    # Transform radar position for this individual point
    azimuth_rad = np.deg2rad(RADAR_AZIMUTH)
    R_az = np.array([
        [np.cos(azimuth_rad), -np.sin(azimuth_rad)],
        [np.sin(azimuth_rad),  np.cos(azimuth_rad)]
    ])

    # Apply rotation and translation to x, y coordinates
    local_xy = np.array([point[0], point[1]])
    rotated_xy = R_az @ local_xy
    global_xy = rotated_xy + np.array([RADAR_X, RADAR_Y])

    # Assign zone based on this point's individual transformed position
    transformed_x, transformed_y = global_xy
    original_z = point[2]
    zone_id = assign_zone_to_point(transformed_x, transformed_y)

    return [transformed_x, transformed_y, original_z, zone_id]

def calculate_frame_statistics(points):
    """
    Calculate various statistics for a point cloud frame.

    Args:
        points: List of [x, y, z, zone_id] coordinates

    Returns:
        dict: Statistics including count, density, distances, etc.
    """
    if len(points) == 0:
        return {
            'num_points': 0,
            'density': 0.0,
            'avg_distance': 0.0,
            'min_distance': 0.0,
            'max_distance': 0.0,
            'std_distance': 0.0,
            'bounding_box_volume': 0.0,
            'unique_zones': 0
        }

    points_array = np.array(points)
    xyz_coords = points_array[:, :3]  # Extract x, y, z coordinates
    zone_ids = points_array[:, 3]     # Extract zone IDs

    num_points = len(points)

    # Calculate bounding box and volume
    if num_points > 1:
        min_coords = np.min(xyz_coords, axis=0)
        max_coords = np.max(xyz_coords, axis=0)
        bbox_dims = max_coords - min_coords
        bounding_box_volume = np.prod(bbox_dims)  # Volume of bounding box

        # Calculate density (points per unit volume)
        density = num_points / max(bounding_box_volume, 1e-8)  # Avoid division by zero
    else:
        density = 0.0
        bounding_box_volume = 0.0

    # Calculate pairwise distances between points
    if num_points > 1:
        distances = pdist(xyz_coords)  # Pairwise distances
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
    else:
        avg_distance = min_distance = max_distance = std_distance = 0.0

    # Count unique zones
    unique_zones = len(np.unique(zone_ids))

    return {
        'num_points': num_points,
        'density': density,
        'avg_distance': avg_distance,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'std_distance': std_distance,
        'bounding_box_volume': bounding_box_volume,
        'unique_zones': unique_zones
    }

def update_global_statistics(global_stats, frame_stats, label):
    """Update global statistics with frame statistics."""
    if 'frames_processed' not in global_stats:
        global_stats['frames_processed'] = 0
        global_stats['by_label'] = {}
        global_stats['overall'] = {
            'num_points': [],
            'density': [],
            'avg_distance': [],
            'min_distance': [],
            'max_distance': [],
            'std_distance': [],
            'bounding_box_volume': [],
            'unique_zones': []
        }

    # Update overall statistics
    for key in global_stats['overall']:
        if key in frame_stats:
            global_stats['overall'][key].append(frame_stats[key])

    # Update label-specific statistics
    if label not in global_stats['by_label']:
        global_stats['by_label'][label] = {
            'count': 0,
            'num_points': [],
            'density': [],
            'avg_distance': [],
            'min_distance': [],
            'max_distance': [],
            'std_distance': [],
            'bounding_box_volume': [],
            'unique_zones': []
        }

    global_stats['by_label'][label]['count'] += 1
    for key in global_stats['by_label'][label]:
        if key != 'count' and key in frame_stats:
            global_stats['by_label'][label][key].append(frame_stats[key])

    global_stats['frames_processed'] += 1

def print_statistics_summary(global_stats):
    """Print a comprehensive summary of all collected statistics."""
    if global_stats['frames_processed'] == 0:
        print("No statistics to display.")
        return

    print("\n" + "=" * 80)
    print("POINT CLOUD STATISTICS SUMMARY")
    print("=" * 80)

    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS ({global_stats['frames_processed']} frames processed)")
    print("-" * 60)

    for stat_name, values in global_stats['overall'].items():
        if values:
            values = np.array(values)
            print(f"{stat_name.replace('_', ' ').title()}:")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Min:  {np.min(values):.3f}")
            print(f"  Max:  {np.max(values):.3f}")
            print(f"  Std:  {np.std(values):.3f}")
            print()

    # Label-specific statistics
    print(f"\n📈 STATISTICS BY ACTION LABEL")
    print("-" * 60)

    for label, label_stats in sorted(global_stats['by_label'].items()):
        if label == -1:
            continue  # Skip invalid labels

        print(f"\nAction: '{label}' ({label_stats['count']} frames)")
        print("~" * 40)

        for stat_name, values in label_stats.items():
            if stat_name == 'count' or not values:
                continue

            values = np.array(values)
            print(f"  {stat_name.replace('_', ' ').title()}:")
            print(f"    Mean: {np.mean(values):.3f}, Min: {np.min(values):.3f}, "
                  f"Max: {np.max(values):.3f}, Std: {np.std(values):.3f}")

    print("\n" + "=" * 80)

def extract_point_clouds_from_json(json_file_path, labels_npy_path):
    with open(json_file_path, 'r') as f:
        try:
            data = json.load(f)
        except:
            print(f"Error loading json file {json_file_path}")
            return None, {}

    print(f"label file: {labels_npy_path}")
    label_data = np.load(labels_npy_path)

    output = []
    global_stats = {}  # Initialize statistics collection
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

            # Process each point individually to get its own zone based on its position
            frame_point_clouds_with_zones = []
            for point in frame_point_clouds:
                processed_point = process_point_with_zone(point)
                frame_point_clouds_with_zones.append(processed_point)

            # Calculate statistics for this frame
            frame_stats = calculate_frame_statistics(frame_point_clouds_with_zones)
            update_global_statistics(global_stats, frame_stats, label)

            # For frame-level zone info (optional), use the most common zone
            if frame_point_clouds_with_zones:
                zones_in_frame = [point[3] for point in frame_point_clouds_with_zones]
                zone_counts = {}
                for z in zones_in_frame:
                    zone_counts[z] = zone_counts.get(z, 0) + 1
                frame_zone = max(zone_counts.items(), key=lambda x: x[1])[0]
            else:
                frame_zone = 0

            output.append({"x": np.asarray(frame_point_clouds_with_zones), "y": label, "timestamp": timestamp_ms, "zone": frame_zone})

    return output, global_stats

def process_scenario_folder(scenario_folder, subject, scenario):
    radar_json_path = os.path.join(scenario_folder, 'radar_1', '6843.json')
    labels_npy_path = os.path.join(scenario_folder, 'ts_repetitions.npy')
    if not os.path.exists(radar_json_path):
        print(f"6843.json not found in {scenario_folder}")
        return {}

    result = extract_point_clouds_from_json(radar_json_path, labels_npy_path)
    if result[0] is None:
        return {}

    scenario_data, scenario_stats = result

    pickle_file_path = os.path.join("data/raw_carelab_zoned_per_point", f"{subject}_{scenario}.pkl")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(scenario_data, f)

    print(f"Saved pickle file for scenario in {scenario_folder}")
    print(f"  → Calculated zones dynamically for {len(scenario_data)} frames")

    # Print brief statistics for this scenario
    if scenario_stats.get('frames_processed', 0) > 0:
        overall_points = scenario_stats['overall']['num_points']
        if overall_points:
            print(f"  → Points per frame: avg={np.mean(overall_points):.1f}, "
                  f"min={np.min(overall_points)}, max={np.max(overall_points)}")

    return scenario_stats

def merge_statistics(combined_stats, scenario_stats):
    """Merge statistics from a scenario into the combined statistics."""
    if not scenario_stats or 'frames_processed' not in scenario_stats:
        return

    if 'frames_processed' not in combined_stats:
        combined_stats['frames_processed'] = 0
        combined_stats['by_label'] = {}
        combined_stats['overall'] = {
            'num_points': [],
            'density': [],
            'avg_distance': [],
            'min_distance': [],
            'max_distance': [],
            'std_distance': [],
            'bounding_box_volume': [],
            'unique_zones': []
        }

    # Merge overall statistics
    for key in combined_stats['overall']:
        if key in scenario_stats['overall']:
            combined_stats['overall'][key].extend(scenario_stats['overall'][key])

    # Merge label-specific statistics
    for label, label_stats in scenario_stats['by_label'].items():
        if label not in combined_stats['by_label']:
            combined_stats['by_label'][label] = {
                'count': 0,
                'num_points': [],
                'density': [],
                'avg_distance': [],
                'min_distance': [],
                'max_distance': [],
                'std_distance': [],
                'bounding_box_volume': [],
                'unique_zones': []
            }

        combined_stats['by_label'][label]['count'] += label_stats['count']
        for key in combined_stats['by_label'][label]:
            if key != 'count' and key in label_stats:
                combined_stats['by_label'][label][key].extend(label_stats[key])

    combined_stats['frames_processed'] += scenario_stats['frames_processed']

def traverse_and_process_folder(root_folder):
    print("=== CARELAB DATA EXTRACTOR WITH DYNAMIC ZONE CALCULATION & STATISTICS ===")
    print(f"Processing data from: {root_folder}")
    print("Zone definitions:")
    for zone in ZONES:
        print(f"  Zone {zone['zone_id']} ({zone['label']}): x[{zone['x_min']:.1f}, {zone['x_max']:.1f}] y[{zone['y_min']:.1f}, {zone['y_max']:.1f}]")
    print(f"Radar config: azimuth={RADAR_AZIMUTH}°, position=({RADAR_X}, {RADAR_Y})")
    print("-" * 80)

    total_processed = 0
    combined_statistics = {}

    for subject in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]:
        for scenario in range(1, 9):
            scenario_folder = os.path.join(root_folder, subject, str(scenario), "regular_session")
            if os.path.isdir(scenario_folder):
                print(f"Processing Subject {subject}, Scenario {scenario}...")
                scenario_stats = process_scenario_folder(scenario_folder, subject, scenario)
                merge_statistics(combined_statistics, scenario_stats)
                total_processed += 1
            else:
                print(f"Scenario folder {scenario_folder} does not exist")

    print("-" * 80)
    print(f"✓ Processing completed: {total_processed} scenario files processed with dynamic zone calculation")

    # Display comprehensive statistics summary
    if combined_statistics:
        print_statistics_summary(combined_statistics)

if __name__ == "__main__":
    root_folder = '/Users/koorosh/Library/CloudStorage/OneDrive-UniversityofToronto/Koorosh-CareLab-Data'
    traverse_and_process_folder(root_folder)