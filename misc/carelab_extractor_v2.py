import os
import csv
import json
import pickle
import numpy as np
import datetime
from scipy.spatial.distance import pdist

# Radar configuration
RADAR_AZIMUTH = 285  # degrees
RADAR_X = 0.14       # radar position x
RADAR_Y = 0.35       # radar position y

# BLE beacon MAC addresses (consistent across all subjects/scenarios)
# Order defines the 3 RSSI channels: [beacon_1, beacon_2, beacon_3]
BLE_BEACONS = [
    'AC:23:3F:AB:CA:2F',  # beacon 1
    'AC:23:3F:AB:CA:A4',  # beacon 2
    'AC:23:3F:F0:95:3A',  # beacon 3
]


# ---------------------------------------------------------------------------
# IMU / BLE loading and timestamp alignment
# ---------------------------------------------------------------------------

def parse_imu_timestamp(ts_str):
    """
    Parse IMU/BLE timestamp string, ignoring the last two fixed fields.
    Format: YYYY-M-DD-HH-MM-SS-<ignored>-<ignored>
    Returns a datetime object at second-level resolution.
    """
    parts = ts_str.strip().split('-')
    # Take only first 6 parts: year, month, day, hour, minute, second
    year, month, day, hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
    return datetime.datetime(year, month, day, hour, minute, second)


def load_imu_csv(csv_path):
    """
    Load an IMU CSV file (acc or gyro).
    First row is header. Columns: timestamp, x, y, z.

    Returns list of (datetime_with_ms, [x, y, z]) tuples,
    where millisecond offsets are inferred from sample count within each second.
    """
    raw_rows = []
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            try:
                ts = parse_imu_timestamp(row[0])
                values = [float(row[1]), float(row[2]), float(row[3])]
                raw_rows.append((ts, values))
            except Exception:
                continue

    if not raw_rows:
        return []

    # Group by second to infer millisecond offsets
    result = []
    i = 0
    while i < len(raw_rows):
        current_sec = raw_rows[i][0]
        # Collect all samples in this second
        group = []
        j = i
        while j < len(raw_rows) and raw_rows[j][0] == current_sec:
            group.append(raw_rows[j])
            j += 1

        # Distribute millisecond offsets evenly within the second
        n_samples = len(group)
        for idx, (ts, values) in enumerate(group):
            ms_offset = int((idx / n_samples) * 1000)
            ts_with_ms = ts.replace(microsecond=ms_offset * 1000)
            result.append((ts_with_ms, values))

        i = j

    return result


def load_ble_csv(csv_path):
    """
    Load a BLE CSV file.
    First row is header. Columns: timestamp, mac_addr, rssi.
    Each timestamp may have multiple rows (one per beacon).

    Pivots into one entry per timestamp with 3 RSSI values ordered by BLE_BEACONS.
    Missing beacon readings for a timestamp are filled with 0.0.

    Returns list of (datetime_with_ms, [rssi_1, rssi_2, rssi_3]) tuples,
    where millisecond offsets are inferred from number of unique timestamps per second.
    """
    # Read all rows grouped by timestamp
    from collections import defaultdict
    raw = defaultdict(dict)  # ts_str -> {mac: rssi}
    ts_order = []            # preserve insertion order of timestamps

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            try:
                ts_str, mac, rssi = row[0].strip(), row[1].strip(), float(row[2])
            except Exception:
                continue
            if ts_str not in raw:
                ts_order.append(ts_str)
            raw[ts_str][mac] = rssi

    if not ts_order:
        return []

    # Parse timestamps and group unique timestamps by second
    parsed = [(ts_str, parse_imu_timestamp(ts_str)) for ts_str in ts_order]

    # Group by second to infer millisecond offsets
    result = []
    i = 0
    while i < len(parsed):
        current_sec = parsed[i][1]
        group = []
        j = i
        while j < len(parsed) and parsed[j][1] == current_sec:
            group.append(parsed[j])
            j += 1

        n_samples = len(group)
        for idx, (ts_str, ts) in enumerate(group):
            ms_offset = int((idx / n_samples) * 1000)
            ts_with_ms = ts.replace(microsecond=ms_offset * 1000)
            rssi_vals = [raw[ts_str].get(mac, 0.0) for mac in BLE_BEACONS]
            result.append((ts_with_ms, rssi_vals))

        i = j

    return result


def build_imu_lookup(imu_data):
    """
    Build arrays for fast nearest-neighbor timestamp lookup.
    Returns (timestamps_ms_array, values_array) where timestamps are in epoch ms.
    """
    if not imu_data:
        return np.array([]), np.array([])

    timestamps = np.array([int(ts.timestamp() * 1000) for ts, _ in imu_data])
    values = np.array([vals for _, vals in imu_data])
    return timestamps, values


def lookup_nearest(ts_ms, lookup_timestamps, lookup_values):
    """
    Find the nearest sample to a given timestamp (in epoch ms).
    Returns the 3-element value array, or zeros if no data available.
    """
    if len(lookup_timestamps) == 0:
        return np.zeros(3)

    idx = np.searchsorted(lookup_timestamps, ts_ms)
    # Check the closest of idx-1 and idx
    if idx == 0:
        return lookup_values[0]
    if idx >= len(lookup_timestamps):
        return lookup_values[-1]

    if abs(lookup_timestamps[idx] - ts_ms) < abs(lookup_timestamps[idx - 1] - ts_ms):
        return lookup_values[idx]
    else:
        return lookup_values[idx - 1]


def load_frame_level_signals(scenario_folder):
    """
    Load IMU (acc + gyro) and BLE signals from a scenario folder.
    Expects: <scenario_folder>/imu/acc.csv, gyro.csv, ble.csv

    Returns three lookup tuples: (acc_ts, acc_vals), (gyro_ts, gyro_vals), (ble_ts, ble_vals)
    Each is (np.array of epoch_ms, np.array of shape (N, 3)).
    """
    imu_folder = os.path.join(scenario_folder, 'imu')

    acc_path = os.path.join(imu_folder, 'acc.csv')
    gyro_path = os.path.join(imu_folder, 'gyro.csv')
    ble_path = os.path.join(imu_folder, 'ble.csv')

    acc_lookup = (np.array([]), np.array([]))
    gyro_lookup = (np.array([]), np.array([]))
    ble_lookup = (np.array([]), np.array([]))

    if os.path.exists(acc_path):
        acc_data = load_imu_csv(acc_path)
        acc_lookup = build_imu_lookup(acc_data)
    else:
        print(f"  WARNING: acc.csv not found at {acc_path}")

    if os.path.exists(gyro_path):
        gyro_data = load_imu_csv(gyro_path)
        gyro_lookup = build_imu_lookup(gyro_data)
    else:
        print(f"  WARNING: gyro.csv not found at {gyro_path}")

    if os.path.exists(ble_path):
        ble_data = load_ble_csv(ble_path)
        ble_lookup = build_imu_lookup(ble_data)
    else:
        print(f"  WARNING: ble.csv not found at {ble_path}")

    return acc_lookup, gyro_lookup, ble_lookup


def get_frame_level_aux(timestamp_ms, acc_lookup, gyro_lookup, ble_lookup):
    """
    Get the frame-level auxiliary signal vector for a given radar frame timestamp.

    Returns: np.array of shape (9,):
        [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ble_1, ble_2, ble_3]
    """
    acc_vals = lookup_nearest(timestamp_ms, acc_lookup[0], acc_lookup[1])
    gyro_vals = lookup_nearest(timestamp_ms, gyro_lookup[0], gyro_lookup[1])
    ble_vals = lookup_nearest(timestamp_ms, ble_lookup[0], ble_lookup[1])

    return np.concatenate([acc_vals, gyro_vals, ble_vals])


# ---------------------------------------------------------------------------
# Point cloud processing (zone removed, density retained)
# ---------------------------------------------------------------------------

def calculate_local_density(point_idx, all_points, radius=0.2, max_density=5.0):
    """
    Calculate local density for a point based on nearby points within a radius.
    Uses logarithmic scaling for better differentiation between high-density areas.

    Args:
        point_idx: Index of the point to calculate density for
        all_points: Array of all transformed points [[x, y, z, doppler, snr], ...]
        radius: Radius within which to count nearby points (default: 0.2m)
        max_density: Maximum density value for normalization (default: 5.0)

    Returns:
        float: Local density value with logarithmic scaling
    """
    if len(all_points) <= 1:
        return 0.0

    target_point = np.array(all_points[point_idx][:3])  # x, y, z coordinates
    nearby_count = 0

    for i, other_point in enumerate(all_points):
        if i == point_idx:
            continue
        other_coords = np.array(other_point[:3])
        distance = np.linalg.norm(target_point - other_coords)
        if distance <= radius:
            nearby_count += 1

    if nearby_count == 0:
        return 0.0
    elif nearby_count <= 5:
        return (nearby_count / 5.0) * (max_density / 2.0)
    else:
        log_density = np.log(nearby_count + 1)
        max_log = np.log(51)
        scaled_density = (log_density / max_log) * max_density
        return min(scaled_density, max_density)


def process_point(point):
    """
    Transform a single radar point (no zone assignment).

    Args:
        point: [x, y, z, doppler, snr] coordinates (before transformation)

    Returns:
        [x, y, z, doppler, snr] coordinates (after transformation)
    """
    azimuth_rad = np.deg2rad(RADAR_AZIMUTH)
    R_az = np.array([
        [np.cos(azimuth_rad), -np.sin(azimuth_rad)],
        [np.sin(azimuth_rad),  np.cos(azimuth_rad)]
    ])

    local_xy = np.array([point[0], point[1]])
    rotated_xy = R_az @ local_xy
    global_xy = rotated_xy + np.array([RADAR_X, RADAR_Y])

    transformed_x, transformed_y = global_xy
    original_z = point[2]
    doppler = point[3]
    snr = point[4]

    return [transformed_x, transformed_y, original_z, doppler, snr]


# ---------------------------------------------------------------------------
# Statistics (updated for new format without zone)
# ---------------------------------------------------------------------------

def calculate_frame_statistics(points):
    """
    Calculate statistics for a point cloud frame.

    Args:
        points: List of [x, y, z, doppler, snr, local_density] coordinates
    """
    if len(points) == 0:
        return {
            'num_points': 0, 'density': 0.0,
            'avg_distance': 0.0, 'min_distance': 0.0, 'max_distance': 0.0, 'std_distance': 0.0,
            'bounding_box_volume': 0.0,
            'avg_local_density': 0.0, 'min_local_density': 0.0, 'max_local_density': 0.0, 'std_local_density': 0.0,
            'avg_snr': 0.0, 'min_snr': 0.0, 'max_snr': 0.0, 'std_snr': 0.0,
            'avg_doppler': 0.0, 'min_doppler': 0.0, 'max_doppler': 0.0, 'std_doppler': 0.0,
        }

    points_array = np.array(points)
    xyz_coords = points_array[:, :3]
    doppler_values = points_array[:, 3]
    snr_values = points_array[:, 4]
    local_densities = points_array[:, 5]

    num_points = len(points)

    if num_points > 1:
        min_coords = np.min(xyz_coords, axis=0)
        max_coords = np.max(xyz_coords, axis=0)
        bbox_dims = max_coords - min_coords
        bounding_box_volume = np.prod(bbox_dims)
        density = num_points / max(bounding_box_volume, 1e-8)
        distances = pdist(xyz_coords)
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
    else:
        density = bounding_box_volume = 0.0
        avg_distance = min_distance = max_distance = std_distance = 0.0

    return {
        'num_points': num_points, 'density': density,
        'avg_distance': avg_distance, 'min_distance': min_distance,
        'max_distance': max_distance, 'std_distance': std_distance,
        'bounding_box_volume': bounding_box_volume,
        'avg_local_density': np.mean(local_densities), 'min_local_density': np.min(local_densities),
        'max_local_density': np.max(local_densities), 'std_local_density': np.std(local_densities),
        'avg_snr': np.mean(snr_values), 'min_snr': np.min(snr_values),
        'max_snr': np.max(snr_values), 'std_snr': np.std(snr_values),
        'avg_doppler': np.mean(doppler_values), 'min_doppler': np.min(doppler_values),
        'max_doppler': np.max(doppler_values), 'std_doppler': np.std(doppler_values),
    }


STAT_KEYS = [
    'num_points', 'density',
    'avg_distance', 'min_distance', 'max_distance', 'std_distance',
    'bounding_box_volume',
    'avg_local_density', 'min_local_density', 'max_local_density', 'std_local_density',
    'avg_snr', 'min_snr', 'max_snr', 'std_snr',
    'avg_doppler', 'min_doppler', 'max_doppler', 'std_doppler',
]


def _empty_stat_lists():
    return {k: [] for k in STAT_KEYS}


def update_global_statistics(global_stats, frame_stats, label):
    """Update global statistics with frame statistics."""
    if 'frames_processed' not in global_stats:
        global_stats['frames_processed'] = 0
        global_stats['by_label'] = {}
        global_stats['overall'] = _empty_stat_lists()

    for key in global_stats['overall']:
        if key in frame_stats:
            global_stats['overall'][key].append(frame_stats[key])

    if label not in global_stats['by_label']:
        global_stats['by_label'][label] = {'count': 0, **_empty_stat_lists()}

    global_stats['by_label'][label]['count'] += 1
    for key in global_stats['by_label'][label]:
        if key != 'count' and key in frame_stats:
            global_stats['by_label'][label][key].append(frame_stats[key])

    global_stats['frames_processed'] += 1


def merge_statistics(combined_stats, scenario_stats):
    """Merge statistics from a scenario into the combined statistics."""
    if not scenario_stats or 'frames_processed' not in scenario_stats:
        return

    if 'frames_processed' not in combined_stats:
        combined_stats['frames_processed'] = 0
        combined_stats['by_label'] = {}
        combined_stats['overall'] = _empty_stat_lists()

    for key in combined_stats['overall']:
        if key in scenario_stats['overall']:
            combined_stats['overall'][key].extend(scenario_stats['overall'][key])

    for label, label_stats in scenario_stats['by_label'].items():
        if label not in combined_stats['by_label']:
            combined_stats['by_label'][label] = {'count': 0, **_empty_stat_lists()}
        combined_stats['by_label'][label]['count'] += label_stats['count']
        for key in combined_stats['by_label'][label]:
            if key != 'count' and key in label_stats:
                combined_stats['by_label'][label][key].extend(label_stats[key])

    combined_stats['frames_processed'] += scenario_stats['frames_processed']


def print_statistics_summary(global_stats):
    """Print a comprehensive summary of all collected statistics."""
    if global_stats['frames_processed'] == 0:
        print("No statistics to display.")
        return

    print("\n" + "=" * 80)
    print("POINT CLOUD STATISTICS SUMMARY")
    print("=" * 80)

    print(f"\nOVERALL STATISTICS ({global_stats['frames_processed']} frames processed)")
    print("-" * 60)

    for stat_name, values in global_stats['overall'].items():
        if values:
            values = np.array(values)
            print(f"{stat_name.replace('_', ' ').title()}:")
            print(f"  Mean: {np.mean(values):.3f}, Min: {np.min(values):.3f}, "
                  f"Max: {np.max(values):.3f}, Std: {np.std(values):.3f}")

    print(f"\nSTATISTICS BY ACTION LABEL")
    print("-" * 60)

    sorted_labels = sorted(global_stats['by_label'].items(), key=lambda x: str(x[0]))
    for label, label_stats in sorted_labels:
        if label == -1:
            continue
        print(f"\nAction: '{label}' ({label_stats['count']} frames)")
        for stat_name, values in label_stats.items():
            if stat_name == 'count' or not values:
                continue
            values = np.array(values)
            print(f"  {stat_name.replace('_', ' ').title()}: "
                  f"Mean={np.mean(values):.3f}, Min={np.min(values):.3f}, "
                  f"Max={np.max(values):.3f}, Std={np.std(values):.3f}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_point_clouds_from_json(json_file_path, labels_npy_path, acc_lookup, gyro_lookup, ble_lookup):
    """
    Extract point clouds from radar JSON and align with IMU/BLE signals.

    Output per frame:
        "point_cloud":    np.array of shape (N_points, 6) — [x, y, z, doppler, snr, local_density]
        "frame_signals":  np.array of shape (9,) — [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ble_1, ble_2, ble_3]
        "label":          action label (str or -1)
        "timestamp":      int (epoch ms)
    """
    with open(json_file_path, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            print(f"Error loading json file {json_file_path}")
            return None, {}

    print(f"  label file: {labels_npy_path}")
    label_data = np.load(labels_npy_path)

    output = []
    global_stats = {}

    for frame in data:
        if 'pointCloud' not in frame:
            continue

        frame_point_clouds = [point_cloud[:5] for point_cloud in frame['pointCloud']]

        frame_timestamp = datetime.datetime.strptime(frame['timeStamp'], "%d-%m-%Y-%H-%M-%S-%f")
        timestamp_ms = int(frame_timestamp.timestamp() * 1000)

        # --- Label matching ---
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

        # --- Process points: transform, compute density (no zone) ---
        transformed_points = [process_point(point) for point in frame_point_clouds]

        points_with_density = []
        for i, point in enumerate(transformed_points):
            local_density = calculate_local_density(i, transformed_points)
            # [x, y, z, doppler, snr, local_density]
            points_with_density.append(point + [local_density])

        # --- Frame-level auxiliary signals (IMU + BLE) ---
        frame_aux = get_frame_level_aux(timestamp_ms, acc_lookup, gyro_lookup, ble_lookup)

        # --- Statistics ---
        frame_stats = calculate_frame_statistics(points_with_density)
        update_global_statistics(global_stats, frame_stats, label)

        output.append({
            "point_cloud": np.asarray(points_with_density),  # (N, 6): [x, y, z, doppler, snr, density]
            "frame_signals": frame_aux,                       # (9,): [acc(3), gyro(3), ble(3)]
            "label": label,
            "timestamp": timestamp_ms,
        })

    return output, global_stats


def process_scenario_folder(scenario_folder, subject, scenario, output_dir):
    radar_json_path = os.path.join(scenario_folder, 'radar_1', '6843.json')
    labels_npy_path = os.path.join(scenario_folder, 'ts_repetitions.npy')

    if not os.path.exists(radar_json_path):
        print(f"  6843.json not found in {scenario_folder}")
        return {}

    # Load frame-level signals
    acc_lookup, gyro_lookup, ble_lookup = load_frame_level_signals(scenario_folder)

    result = extract_point_clouds_from_json(
        radar_json_path, labels_npy_path,
        acc_lookup, gyro_lookup, ble_lookup
    )
    if result[0] is None:
        return {}

    scenario_data, scenario_stats = result

    os.makedirs(output_dir, exist_ok=True)
    pickle_file_path = os.path.join(output_dir, f"{subject}_{scenario}.pkl")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(scenario_data, f)

    print(f"  Saved: {pickle_file_path} ({len(scenario_data)} frames)")

    # Report alignment stats
    has_acc = acc_lookup[0].shape[0] > 0
    has_gyro = gyro_lookup[0].shape[0] > 0
    has_ble = ble_lookup[0].shape[0] > 0
    print(f"  Frame-level signals: acc={'OK' if has_acc else 'MISSING'}, "
          f"gyro={'OK' if has_gyro else 'MISSING'}, "
          f"ble={'OK' if has_ble else 'MISSING'}")

    if scenario_stats.get('frames_processed', 0) > 0:
        overall_points = scenario_stats['overall']['num_points']
        if overall_points:
            print(f"  Points per frame: avg={np.mean(overall_points):.1f}, "
                  f"min={np.min(overall_points)}, max={np.max(overall_points)}")

    return scenario_stats


def traverse_and_process_folder(root_folder, output_dir="/cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data-Processed"):
    print("=== CARELAB DATA EXTRACTOR V2 (with IMU + BLE, no zone) ===")
    print(f"Input:  {root_folder}")
    print(f"Output: {output_dir}")
    print(f"Radar config: azimuth={RADAR_AZIMUTH}, position=({RADAR_X}, {RADAR_Y})")
    print(f"Point-level features: [x, y, z, doppler, snr, local_density]")
    print(f"Frame-level features: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ble_1, ble_2, ble_3]")
    print("-" * 80)

    total_processed = 0
    combined_statistics = {}

    for subject in [str(i) for i in range(1, 21)]:
        for scenario in range(1, 9):
            scenario_folder = os.path.join(root_folder, subject, str(scenario), "regular_session")
            if os.path.isdir(scenario_folder):
                print(f"Processing Subject {subject}, Scenario {scenario}...")
                scenario_stats = process_scenario_folder(scenario_folder, subject, scenario, output_dir)
                merge_statistics(combined_statistics, scenario_stats)
                total_processed += 1
            else:
                print(f"  Skipping: {scenario_folder} does not exist")

    print("-" * 80)
    print(f"Processing completed: {total_processed} scenarios processed")

    if combined_statistics:
        print_statistics_summary(combined_statistics)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CareLab Data Extractor V2 (with IMU + BLE)")
    parser.add_argument("--input", type=str, required=True, help="Root folder of raw CareLab data")
    parser.add_argument("--output", type=str, default="/cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data-Processed", help="Output directory for pickle files")
    args = parser.parse_args()
    traverse_and_process_folder(args.input, args.output)
