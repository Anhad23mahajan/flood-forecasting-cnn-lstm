# Basic libraries
import os
import glob
import time
import shutil

# Numbers, tables and plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Map files and the dataset download
import rasterio
from huggingface_hub import snapshot_download, HfApi
