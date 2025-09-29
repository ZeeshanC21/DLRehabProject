import os
import wget
import zipfile

# Make sure you're in the right directory
os.makedirs("data/raw/REHAB24-6", exist_ok=True)
os.chdir("data/raw/REHAB24-6")

# Files to download
urls = [
    "https://zenodo.org/records/13305826/files/Segmentation.csv",
    "https://zenodo.org/records/13305826/files/Segmentation.txt",
    "https://zenodo.org/records/13305826/files/joints_names.txt",
    "https://zenodo.org/records/13305826/files/marker_names.txt",
]

# Download each file
for url in urls:
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        wget.download(url, filename)
    else:
        print(f"{filename} already exists, skipping.")

# Unzip everything
for file in os.listdir():
    if file.endswith(".zip"):
        print(f"\nUnzipping {file}...")
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(".")
