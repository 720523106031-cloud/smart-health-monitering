"""
Smart Health Monitoring System - Local Development Runner
Run this file directly with:
    python app.py

Access application at:
    http://127.0.0.1:5000
"""

import sys
import os

# Add root and api directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'api'))

from api.index import app

if __name__ == '__main__':
    print("="*60)
    print("SMART HEALTH MONITORING SYSTEM - FLASK SERVER")
    print("AI-Powered Blood Sugar and Blood Pressure Monitoring")
    print("ECE DevOps College Assignment")
    print("="*60)
    print("Server running at: http://127.0.0.1:5000")
    print("Press Ctrl+C to quit.")
    app.run(host='0.0.0.0', port=5000, debug=True)

