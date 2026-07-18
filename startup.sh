#!/usr/bin/env bash
# Startup script for Azure App Service (Linux, Python).
# App Service passes $PORT for the HTTP port.
exec uvicorn serve:app --host 0.0.0.0 --port "${PORT:-8000}"
