import numpy as np

def read_npy_file(file_path):
    try:
        data = np.load(file_path)
        print("Contents of the npy file:")
        print(data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = "C:\\Users\\Koorosh\\OneDrive - University of Toronto\\Koorosh-CareLab-Data\\1\\8\\regular_session\\ts_repetitions.npy"
    read_npy_file(file_path)