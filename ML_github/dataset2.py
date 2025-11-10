import cv2
import os
import re
import shutil

# ---------------- SETTINGS -----------------
videos_folder = "videos" # Folder containing all your videos
# IMPORTANT: This will be the main folder containing ONLY your class folders.
output_folder = "dataset_classes" 
frame_interval = 1 # Save every nth frame

# Define the final class name (caption) for each video source name.
# --- UPDATED MAPPING BASED ON USER REQUEST ---
# Cafeteria 1, Cafeteria 2, Old Academic Block, and Hostel 9 are now SEPARATE classes.
video_groups_to_class = {
    # UNIQUE CLASSES
    "Basketball": "Basketball",
    "Football": "Football_Ground",
    "Jaggu shop": "Jaggu_Shop",
    "Library": "Library",
    "NAB LAB": "NAB_Lab",
    
    # NOTE: Kailash Area is already merged here by assigning the same target class name.
    "Kailash area part 2": "Kailash_Area",
    "Kailash area": "Kailash_Area", 
    
    # NEW SEPARATE CLASSES
    "Cafe": "Cafeteria_1_Block", # Treating 'Cafe' as part of Cafeteria 1 area
    "Cafetera 1": "Cafeteria_1_Block", 
    "Cafeteria 1 part 2": "Cafeteria_1_Block", 
    
    "Cafeteria 2": "Cafeteria_2_Block", 
    "cafeteria 2 part 2": "Cafeteria_2_Block", 
    "Cafeteria 2 part 3": "Cafeteria_2_Block", 
    
    "Hostel 9 part2": "Hostel_9",
    "hostel 9": "Hostel_9",
    
    "OLd acad bloack 3": "Old_Academic_Block",
    "old acad block 2": "Old_Academic_Block",
    "old acad block 4": "Old_Academic_Block",
    "old acad block 5": "Old_Academic_Block",
    "old acad block": "Old_Academic_Block"
}

# --------------------------------------------

def clean_class_name(name):
    """Cleans up the class name for use as a folder name."""
    # Replace spaces, dots, and hyphens with underscores
    name = re.sub(r'[.\- ]', '_', name)
    # Remove any character that is not alphanumeric or an underscore
    name = re.sub(r'[^\w]', '', name)
    # Convert to lowercase for uniformity
    return name.lower()

# --------------------------------------------

# 1. Clean up or create the main output folder
if os.path.exists(output_folder):
    print(f"Clearing old output folder: {output_folder}")
    # WARNING: This deletes all content in dataset_classes. 
    # Only run if you are sure you want to regenerate the dataset.
    shutil.rmtree(output_folder) 
os.makedirs(output_folder)
print(f"Created new class output folder: {output_folder}")

# Loop through all videos
for video_name in os.listdir(videos_folder):
    if not video_name.endswith(".mp4"):
        continue

    video_path = os.path.join(videos_folder, video_name)
    video_id = os.path.splitext(video_name)[0]  # filename without extension
    
    # 2. Determine the final class folder name
    target_class_name = video_groups_to_class.get(video_id, None)
    
    if target_class_name is None:
        print(f"Skipping {video_name}, no class mapping found.")
        continue
    
    # Clean the class name for the folder
    cleaned_class_folder = clean_class_name(target_class_name)
    class_folder_path = os.path.join(output_folder, cleaned_class_folder)
    
    # 3. Create the class folder if it doesn't exist
    if not os.path.exists(class_folder_path):
        os.makedirs(class_folder_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_name}")
        continue
        
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Create a unique image name combining video ID and frame count
            # This prevents naming clashes when moving files from different videos 
            # into the same class folder (e.g., Cafeteria 1 and Cafeteria 2)
            img_name = f"{video_id.replace(' ', '_')}_frame_{frame_count}.jpg"
            img_path = os.path.join(class_folder_path, img_name)
            
            # Save the frame directly into the class folder
            cv2.imwrite(img_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Processed {video_name} -> Class '{cleaned_class_folder}', saved {saved_count} frames.")

print("\n--- Dataset Generation Complete ---")
print(f"Your new, clean dataset is ready in the '{output_folder}' folder.")
print("Remember to use the flow_from_directory model code now!")