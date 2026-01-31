
import kagglehub
import shutil
import os

print("Downloading dataset 'zlatan599/lexor-2026'...")

# Download latest version
try:
    path = kagglehub.dataset_download("zlatan599/lexor-2026")
    print("Download completed.")
    print("Path to dataset files:", path)
    
    # Optional: Move to workspace if needed (commented out for safety)
    # target_dir = os.path.join(os.getcwd(), 'docs&tests&data_sets', 'data_set', 'lexor-2026')
    # if not os.path.exists(target_dir):
    #     shutil.copytree(path, target_dir)
    #     print(f"Copied to workspace: {target_dir}")
        
except Exception as e:
    print(f"Error downloading dataset: {e}")
