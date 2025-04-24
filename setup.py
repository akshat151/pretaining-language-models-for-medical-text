import os
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.append(project_root)

# Change the working directory to the project root
os.chdir(project_root)

print("Environment set up: sys.path updated, working dir set to project root.")
