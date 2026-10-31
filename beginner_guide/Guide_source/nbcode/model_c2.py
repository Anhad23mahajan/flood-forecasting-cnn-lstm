# Basic libraries
import os
import json
import gc

# Numbers, tables and plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Averaging filter (used for the ground height) and the confusion matrix plot
from scipy.ndimage import uniform_filter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Deep learning
import tensorflow as tf
import keras
from keras import layers

# Google Drive
from google.colab import drive
