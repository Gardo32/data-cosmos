# Gunicorn configuration file
import multiprocessing
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server socket
bind = "0.0.0.0:" + os.environ.get("PORT", "5000")
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 2

# Server mechanics
daemon = False
raw_env = []

# Logging
errorlog = '-'
loglevel = 'info'
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'biopixel-gunicorn'

# Server hooks
def on_starting(server):
    server.log.info("Starting Biopixel Vegetation Analysis server")

def on_exit(server):
    server.log.info("Shutting down Biopixel Vegetation Analysis server")
