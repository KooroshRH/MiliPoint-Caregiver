import numpy as np

def read_npy_file(file_path):
    try:
        data = np.load(file_path)
        print("Contents of the npy file:")
        print(data.shape)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = "data\\raw\\action_label.npy"
    read_npy_file(file_path)