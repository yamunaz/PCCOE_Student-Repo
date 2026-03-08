# This file serves as the entry point for a WSGI server.
# It imports the Flask application instance 'app' from your main 'app.py' file.
from app import app

if __name__ == "__main__":
    # This allows you to run the app directly with 'python wsgi.py' for testing
    app.run()