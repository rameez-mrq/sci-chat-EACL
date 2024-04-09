from start_server import app as application
from flask_cors import CORS
import torch
CORS(application)

if __name__ == "__main__":
    # set_start_method('spawn')
    torch.multiprocessing.set_start_method('spawn', force = True)
    application.run()