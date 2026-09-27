import os
import urllib.request

def download_random_images():
    # Download 8 images for train, 2 for val
    train_dir = "dataset/train/not_onion"
    val_dir = "dataset/val/not_onion"
    
    print("Downloading 'not_onion' images...")
    
    for i in range(8):
        # We use picsum.photos which returns a random image each time
        url = "https://picsum.photos/400/400"
        urllib.request.urlretrieve(url, os.path.join(train_dir, f"random_{i}.jpg"))
        
    for i in range(2):
        url = "https://picsum.photos/400/400"
        urllib.request.urlretrieve(url, os.path.join(val_dir, f"random_{i}.jpg"))
        
    print("Downloaded 10 random 'not_onion' images!")

if __name__ == "__main__":
    download_random_images()
