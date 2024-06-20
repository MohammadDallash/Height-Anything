#!/bin/bash

# Define the URLs of the models
DEPTH_ANYTHING_MODEL_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true"
YOLOV9_MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c-seg.pt"

# Define the directory to save the models
PRETRAINED_WEIGHTS_DIR="pretrained_weights"

# Create the directory if it doesn't exist
mkdir -p $PRETRAINED_WEIGHTS_DIR

# Function to check if a file exists
file_exists() {
    FILE=$1
    [ -f "$FILE" ]
}

# Function to download a model
download_model() {
    URL=$1
    FILENAME=$(basename $URL)
    FILE_PATH="$PRETRAINED_WEIGHTS_DIR/$FILENAME"
    if file_exists "$FILE_PATH"; then
        echo "$FILENAME already exists"
    else
        echo "Downloading $FILENAME..."
        wget --progress=bar:force -O "$FILE_PATH" $URL
        echo "Downloaded $FILENAME"
    fi
}

# Check if the models exist before downloading
if ! file_exists "$PRETRAINED_WEIGHTS_DIR/depth_anything_v2_metric_vkitti_vitl.pth"; then
    download_model $DEPTH_ANYTHING_MODEL_URL
fi

if ! file_exists "$PRETRAINED_WEIGHTS_DIR/yolov9c-seg.pt"; then
    download_model $YOLOV9_MODEL_URL
fi
