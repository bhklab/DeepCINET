import os
import dotenv

APP_ROOT = os.path.join(os.path.dirname(__file__), "..")

dotenv.load_dotenv(os.path.join(APP_ROOT, ".env"), verbose=True)

# Append APP_ROOT to DATA* environment variables
for k, v in list(os.environ.items()):
    if str(k).startswith('DATA'):
        os.environ[k] = os.path.abspath(os.getenv(k))
