import os
import sys
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_APP_PATH = os.path.join(BASE_DIR, "project", "backend", "app.py")

spec = importlib.util.spec_from_file_location("backend_flask_app", BACKEND_APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load backend Flask app module")
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)

app = backend_module.app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)


