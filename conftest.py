import os
import sys

# Workaround for VS Code test discovery missing Conda DLL paths on Windows
if sys.platform == "win32":
    conda_bin = r"D:\anaconda_envs\radar-sim\Library\bin"
    if os.path.exists(conda_bin):
        try:
            # Python 3.8+ requires explicit DLL directory registration
            os.add_dll_directory(conda_bin)
        except OSError:
            pass