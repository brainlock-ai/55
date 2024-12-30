import os
import sys

# Get the absolute path of auto_update.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import auto_update

if __name__ == "__main__":
    updater = auto_update.AutoUpdate()
    updater.start()
    updater.join()  # Keep the main thread alive 