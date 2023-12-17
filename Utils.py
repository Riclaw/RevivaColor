import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

def histogram_equalization(img):
    '''
    Args:
      img (numpy array): image to be equalized
    Returns:
      equalized_img (numpy array): equalized image
    '''
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)  # convert from RGB color-space to YCrCb
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])  # equalize the histogram of the Y channel
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)  # convert back to RGB color-space from YCrCb
    return equalized_img

def white_balance(img):
    '''
    Performs the white balancing of the input image.
    Args:
        img (np.array): image to be balanced
    Returns:
        img (np.array)
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # Convert to LAB using cv2
    avg_a = np.average(img[:, :, 1])    # Mean value of the A channel
    avg_b = np.average(img[:, :, 2])    # Mean value of the B channel
    img[:, :, 1] = img[:, :, 1] - ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.1)    # Balancing the A channel
    img[:, :, 2] = img[:, :, 2] - ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.1)    # Balancing the B channel 
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)  # Convert back to RGB using cv2
    return img

def load_images(folder, resize=True, size=(256, 256), balance=False, lab=False):
    '''
    Args:
      folder (string): path to the folder containing the images
      resize (bool): resize the images or not
      size (tuple): size of the images
      lab (bool): convert the images from RGB to LAB color-space or not
    Returns:
      data (numpy array): array containing the images
    '''
    data = []
    for file in os.listdir('Dataset'): # Load images from the 'Dataset' directory and create a numpy array
        img = np.array(Image.open('Dataset/' + file))
        if resize:  # Resize the image if needed
            img = cv2.resize(img, size)
        if balance: # White-Balance the image if needed
            img = white_balance(img)
        if lab: # Convert the image from RGB to LAB color-space
            img = rgb2lab(img)
        data.append(img)
    return np.array(data, dtype = np.float32)

def plot_images(data, num_img):
    '''
    Args:
      data (numpy array): images to be plotted
      num_img (int): number of images to be plotted
    Returns:
      Plot: num_img images from the data array
    '''
    fig, axes = plt.subplots(2, num_img//2, figsize = (10,4))
    for i in range(num_img):
        ax = axes[i%2, i%num_img//2]
        ax.imshow(np.clip(data[i], 0, 255))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_history(history):
  '''
  Args:
    history (dict): history of training epochs
  Returns:
    Plot:
      - comparison between train and validation accuracy
      - comparison between train and validation loss
  '''
  fig, axes = plt.subplots(1, 2, figsize = (8,4))
  ax, ax1 = axes[0], axes[1]
  ax.plot(history['loss'], alpha=.3, color='#ff7f0e', linestyle='--', label='Train Loss')
  ax.plot(history['val_loss'], alpha=.8, color='#ff7f0e', label='Val Loss',)
  ax.legend(loc='upper right')
  ax.set_title('Loss')

  ax1.plot(history['accuracy'], alpha=.3, color='#055c1d', linestyle='--', label = 'Train Accuracy')
  ax1.plot(history['val_accuracy'], alpha=.8, color='#055c1d', label='Val Accuracy',)
  ax1.legend(loc='upper left')
  ax1.set_title('Accuracy')

  plt.grid(alpha=.3)
  plt.show()

def extract_single_dim_from_LAB_convert_to_RGB(image, idim):
    '''
    Extracts one channel from a LAB image and converts it to RGB
    Args:
        image (np.array): lab image
        idim (int): dimension (0-2) to be extracted
    Returns
        (z): specified channel of the input image in RGB 
    '''
    z = np.zeros(image.shape)
    if idim != 0 :
        z[:,:,0] = 80 # Brightness to plot the image along 1st or 2nd axis
    z[:,:,idim] = image[:,:,idim]
    z = lab2rgb(z)
    return (z)

def plot_lab_spectrums(img1, img2):
    '''
    Decomposes the input images and plots separately the color channels to compare them
    Args:
        img1 (np.array): original image
        img2 (np.array): reconstructed image
    Returns:
        plots
    '''
    lab1 = rgb2lab(img1)
    # lab_l1 = extract_single_dim_from_LAB_convert_to_RGB(lab1,0)
    lab_a1 = extract_single_dim_from_LAB_convert_to_RGB(lab1,1)
    lab_db1 = extract_single_dim_from_LAB_convert_to_RGB(lab1,2)

    lab2 = rgb2lab(img2)
    # lab_l2 = extract_single_dim_from_LAB_convert_to_RGB(lab2,0)
    lab_a2 = extract_single_dim_from_LAB_convert_to_RGB(lab2,1)
    lab_db2 = extract_single_dim_from_LAB_convert_to_RGB(lab2,2)

    data = [('Original Photo', img1), ('Original A', lab_a1), ('Original B', lab_db1), ('Reconstructed A', lab_a2), ('Reconstructed B', lab_db2)]

    fig, axes = plt.subplots(ncols=len(data), figsize=(12, 3))
    for ax, (title, img) in zip(axes, data):
        ax.set_title(title)
        ax.imshow(img)
        ax.axis('off')
    fig.tight_layout()
    plt.show()

def save_history(history, name):
    '''
    Args:
      history (dict): history of training epochs
      name (string): name of the file to be saved

    Returns:
      CSV file: history saved as a CSV file
    '''
    history_df = pd.DataFrame.from_dict(history)

    try:
      history_df.to_csv(f'{name}.csv')
      print('History successfully saved!')

    except:
      print('ERROR while saving the history')

def min_max_prep(X_train, X_val, X_test):
  '''
  This function is going to preprocess all the data
  Args:
    train, validation, test set Xs and Ys
  Returns:
    min max normalized dataset
  '''
  # Calculate the maximum and minimum values for features in the training set
  max_df = X_train.max()
  min_df = X_train.min()

  # Normalize the features for all datasets (training, validation, and test)
  X_train = (X_train - min_df) / (max_df - min_df)
  X_val = (X_val - min_df) / (max_df - min_df)
  X_test = (X_test - min_df) / (max_df - min_df)

  return X_train, X_val, X_test