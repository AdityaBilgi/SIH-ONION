# Onion Quality ML Pipeline

Since you don't have a dataset yet, we are going to build the **Machine Learning Pipeline** using YOLOv8. The code will be ready to go, and I'll show you how to easily create a mini-dataset (just 10-20 images) to test it out!

## Step 1: Create a Mini-Dataset

In real-world applications, especially agricultural grading, the best results come from custom data that matches the lighting and camera angles of your actual procurement centers.

1.  **Gather Images:** Use your phone to take about 10-20 pictures of onions. Make sure to include:
    *   Good quality onions.
    *   Damaged or rotten onions.
    *   Sprouted onions.
2.  **Label the Images (Free Tool):**
    *   Go to [Roboflow](https://roboflow.com/) and create a free account.
    *   Create a new project (Object Detection).
    *   Upload your images.
    *   Use their web tool to draw bounding boxes around the onions. Assign labels like `good`, `damaged`, `rotten`, `sprouted`.
3.  **Export the Dataset:**
    *   Once labeled, click "Generate" to create a dataset version.
    *   Click "Export" and choose the **YOLOv8 format**. You will get a download link or a zip file containing a `data.yaml` file and folders for `train` and `valid` images.

## Step 2: Set up the Environment

We'll use Python and the `ultralytics` library (which provides YOLOv8).

```bash
# Inside this onion_app folder, create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Install the required libraries
pip install ultralytics opencv-python
```

## Step 3: Train the Model

Once you have downloaded your YOLOv8 dataset from Roboflow, extract it into this folder. It should have a `data.yaml` file.

Run the training script I've prepared for you:
```bash
python train.py
```

This script will take your small dataset, train a YOLO model to recognize the defects, and output the model weights which we will later integrate into your mobile app!
