# Fix for TensorFlow/Keras Error

## Problem
The error occurs because:
- You have TensorFlow installed with Keras 3
- The transformers library requires the older `tf-keras` package
- Even with environment variables set, transformers tries to load TensorFlow

## Solutions (Choose ONE)

### ✅ **Option 1: Install tf-keras (Recommended)**
```powershell
pip install tf-keras
```
Then run:
```powershell
python app.py
```

### ✅ **Option 2: Use PyTorch-only transformers**
```powershell
# Uninstall TensorFlow
pip uninstall tensorflow tensorflow-intel -y

# Reinstall transformers (will use PyTorch only)
pip install --upgrade --force-reinstall transformers

# Make sure PyTorch is installed
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then run:
```powershell
python app.py
```

### ✅ **Option 3: Run without ML features (Quick Start)**
If you just want to test the basic app without ML features:

```powershell
# Set environment variable to skip ML imports
$env:SKIP_ML_FEATURES="1"
python app.py
```

Note: This will disable the advanced ML features but the app will run with traditional matching.

## After Fixing, Run the App

```powershell
cd C:\Users\chait\OneDrive\Desktop\SkillSync\SkillSync
python app.py
```

The app will start on:
- **Local:** http://localhost:7860
- **Network:** http://0.0.0.0:7860

## Recommended: Option 1 (tf-keras)

The easiest and fastest fix:
```powershell
pip install tf-keras
python app.py
```

This installs the compatibility layer without changing your existing setup.

## Verification

Once running, you should see:
```
[INFO] Database schema initialized
[INFO] Advanced ML features loaded successfully
* Running on http://0.0.0.0:7860
```

## Test Credentials

**Intern:**
- Email: `alice.smith@example.com`
- Password: `password`

**Recruiter:**
- Email: `emma.wilson@techcorp.com`  
- Password: `password`
