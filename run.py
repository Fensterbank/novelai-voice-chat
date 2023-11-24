import os
import sys
import subprocess

# Disabling output buffering so that the status window can be updated in real time
os.environ['PYTHONUNBUFFERED'] = '1'

print('Starting NovelAI Voice Chat...')
subprocess.run([sys.executable, os.path.join('src', 'main.py')] + sys.argv[1:])
