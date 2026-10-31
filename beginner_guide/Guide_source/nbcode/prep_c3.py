# Basic libraries
import os
import glob
import json
import time

# Numbers, tables and plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Averaging filter (used to find valleys)
from scipy.ndimage import uniform_filter

# Map files and the dataset download
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from huggingface_hub import snapshot_download, HfApi

# Google Drive (only read from, at the end)
from google.colab import drive
