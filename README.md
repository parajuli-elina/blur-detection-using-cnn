# Label Blur Detection using CNN

This project implements a Convolutional Neural Network (CNN) to classify images as either **blur** or **focus**. The workflow includes data loading, preprocessing, augmentation, model training, evaluation, and visualization.

## Project Structure

- **label_blur_detection.ipynb**: Main Jupyter notebook containing all code for data processing, model building, training, evaluation, and visualization.
- **label_detection/Images/**: Directory containing the input images.
- **dataset/metadata.csv**: CSV file mapping image filenames to their labels (0 for blur, 1 for focus).
- **cnn_model/**: Directory where trained model checkpoints are saved.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- OpenCV
- Matplotlib
- scikit-learn
- streamlit (to test the model from UI)

Install dependencies with:

```sh
pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn
```

## Usage

### Prepare Data

Place your images in the `dataset/Images/` directory and ensure `metadata.csv` contains the correct filename-label mapping.

### Run the Notebook

Open `label_blur_detection.ipynb` and execute all cells. The notebook will:

- Load and preprocess images and labels
- Visualize sample images
- Split data into training and validation sets
- Apply data augmentation
- Build and train a CNN model
- Save model checkpoints
- Plot training/validation accuracy and loss
- Display predictions on validation images

### Model Checkpoints

Model checkpoints are saved in the `cnn_model/` directory after each epoch. The best model can be loaded for inference.

### Model Architecture

- Input rescaling layer
- 3 convolutional blocks (Conv2D + MaxPooling)
- Global average pooling
- 4 dense layers
- Output layer with softmax activation

## Applications

This blur detection model can be applied in various domains, including:

- **CCTV Surveillance:** Automatically identify and filter out blurred frames to ensure only clear footage is used for monitoring and analysis.
- **Satellite Imaging:** Detect and flag blurred satellite images to improve the quality of remote sensing data and mapping.
- **Medical Imaging:** Identify blurred medical images (such as X-rays, MRIs, or CT scans) to prevent diagnostic errors and ensure only high-quality images are used for clinical evaluation.
