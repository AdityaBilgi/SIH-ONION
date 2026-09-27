import os
import shutil
import random

def setup_classification_dataset():
    base_dir = "dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    # Create train and val directories
    for category in ["good", "bad"]:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        
        # Get all images for this category
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        images = [f for f in os.listdir(category_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        # Split (leave 1 for validation)
        val_images = images[:1]
        train_images = images[1:]
        
        # Move images
        for img in train_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(train_dir, category, img))
        for img in val_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(val_dir, category, img))
            
    print("Dataset structured for YOLOv8 Classification!")

if __name__ == "__main__":
    setup_classification_dataset()
