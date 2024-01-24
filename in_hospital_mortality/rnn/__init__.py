import sys
import os

# Get the directory of the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to the grandparent folder
grandparent_folder = os.path.dirname(os.path.dirname(current_script_path))

# Add the grandparent folder to sys.path
sys.path.append(grandparent_folder)