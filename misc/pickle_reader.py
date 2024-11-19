import pickle
import pandas as pd

def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def summarize_data(data):
    if isinstance(data, pd.DataFrame):
        print("Data Summary:")
        print(data.describe())
        print("\nData Info:")
        print(data.info())
    else:
        print("Data Type:", type(data))
        print("Data Content:", data)

def main():
    file_path = input("Enter the path to the pickle file: ")
    data = read_pickle(file_path)
    print("Data Content:")
    print(data)
    print("\nSummary:")
    summarize_data(data)

if __name__ == "__main__":
    main()