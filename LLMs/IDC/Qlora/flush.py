#Delete all files in given folder

import os
import sys

def delete_files_in_folder(folder_path):
    try:
        # Check if the provided path is a directory
        if not os.path.isdir(folder_path):
            print("Error: The provided path is not a directory.")
            return

        # Get a list of all files in the directory
        files = os.listdir(folder_path)

        # Iterate through the files and delete them
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print("All files in the folder have been deleted.")

    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    # Check if the folder path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py folder_path")
    else:
        folder_path = sys.argv[1]
        delete_files_in_folder(folder_path)