# TensorFlow Image Classification

This project uses a pre-trained VGG16 model to classify images. The model is fine-tuned and exported for use with TensorFlow.js.

## Prerequisites

- Docker
- Docker Compose

## Setup Instructions

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Build and start the Docker container

```bash
docker-compose up -d
```

### Step 3: Access the Docker container

Find the container ID for the TensorFlow container:

```bash
docker ps
```

Then, access the container:

```bash
docker exec -it <CONTAINER ID> bash
```

### Step 4: Install Python dependencies

Upgrade `pip` and install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Prepare your dataset

Place your training images in the `images` directory. The directory should have subdirectories for each class of images. For example:

```
images/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

### Step 6: Train the model

```bash
python train.py
```

This will train the model and save it in TensorFlow.js format in the `tfjs_model` directory.

## File Structure

- `train.py`: The script for training the model.
- `docker-compose.yml`: Docker Compose configuration file.
- `requirements.txt`: Python dependencies file.
- `images/`: Directory containing your training images.
- `tfjs_model/`: Directory where the trained model in TensorFlow.js format will be saved.

## Additional Notes

- Ensure your images are properly labeled and placed in the correct directories.
- Adjust the `image_size` and `batch_size` parameters as needed for your specific use case.
- You can modify the model architecture by changing the layers added to `base_model` in `train.py`.

This project is set up to use TensorFlow with GPU support. Make sure you have the necessary drivers and CUDA installed if you plan to use GPU acceleration.
```