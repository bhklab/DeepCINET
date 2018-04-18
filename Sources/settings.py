import os
import dotenv

APP_ROOT = os.path.join(os.path.dirname(__file__), "..")

dotenv.load_dotenv(os.path.join(APP_ROOT, ".env"), verbose=True)

# Append APP_ROOT to DATA* environment variables
for k, v in list(os.environ.items()):
    if str(k).startswith('DATA'):
        os.environ[k] = os.path.abspath(os.getenv(k))

# At least there should be an image rotation
IMAGE_ROTATIONS = {
    'x': 1,
    'y': 1,
    'z': 4
}

TOTAL_ROTATIONS = IMAGE_ROTATIONS['x']*IMAGE_ROTATIONS['y']*IMAGE_ROTATIONS['z']


