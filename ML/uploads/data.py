import os
import cv2
import pandas as pd

def create_csv_from_folders(dataset_path, csv_path):
    # List to hold data (image path and corresponding label)
    data = []
    
    # Loop through each folder in the dataset path
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        
        # Ensure that it's a folder and the folder name is numeric (indicating age or label)
        if os.path.isdir(folder_path) and folder_name.isdigit():
            label = int(folder_name)  # Folder name is the label
            
            # Loop through each image in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                # Check if the image file is a valid image type (e.g., jpg, jpeg, png)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append([image_path, label])
    
    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=['Image_Path', 'Label'])
    
    # Save the DataFrame to CSV
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved at {csv_path}")

# Example usage
dataset_path = r'D:/Be/BE/ML/uploads/face_age'  # Path to your dataset folder
csv_path = r'D:/Be/BE/ML/uploads/face_age_labels.csv'  # Path where the CSV will be saved

create_csv_from_folders(dataset_path, csv_path)
