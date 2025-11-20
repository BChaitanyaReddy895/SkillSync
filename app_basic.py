"""
Simple version of app.py that runs WITHOUT advanced ML features
Use this for quick testing or if you have import issues
"""
import os

# Disable ALL ML features before any imports
os.environ['SKIP_ML_FEATURES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

print("=" * 50)
print("SkillSync - Running in BASIC MODE")
print("Advanced ML features are DISABLED for quick start")
print("=" * 50)
print()

# Now import and run the main app
exec(open('app.py').read())
