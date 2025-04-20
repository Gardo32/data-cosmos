#!/bin/bash
set -e

# Create necessary directories
mkdir -p /data/uploads
mkdir -p /data/analysis

# Ensure proper permissions
chmod -R 777 /data

# Start Gunicorn server with configuration file
exec gunicorn --config /app/gunicorn.conf.py app:app "$@"