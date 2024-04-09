from start_server import app as application
# from blender_start_server import app as application
from flask_cors import CORS
import torch
CORS(application)

if __name__ == "__main__":
    # set_start_method('spawn')
    torch.multiprocessing.set_start_method('spawn', force = True)
    application.run()
# #uWSGI @ file:///home/conda/feedstock_root/build_artifacts/uwsgi_1697223254948/work