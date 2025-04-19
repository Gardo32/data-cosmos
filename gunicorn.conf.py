import multiprocessing
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', 1000))
max_requests_jitter = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', 50))

log_file = os.environ.get('GUNICORN_LOG_FILE', "-")

bind = f"{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', 10000)}"

workers = int(os.environ.get('GUNICORN_WORKERS', (multiprocessing.cpu_count() * 2) + 1))
threads = int(os.environ.get('GUNICORN_THREADS', workers))
worker_class = os.environ.get('GUNICORN_WORKER_CLASS', 'gthread')

timeout = int(os.environ.get('GUNICORN_TIMEOUT', 120))
