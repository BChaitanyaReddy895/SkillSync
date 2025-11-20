# SkillSync Quick Start Script
# This script fixes the TensorFlow/Keras compatibility issue and runs the app

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SkillSync - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

Write-Host "Checking Python..." -ForegroundColor Yellow
python --version

Write-Host ""
Write-Host "Choose how to fix the TensorFlow/Keras issue:" -ForegroundColor Green
Write-Host "  [1] Install tf-keras (RECOMMENDED - fastest)" -ForegroundColor White
Write-Host "  [2] Remove TensorFlow, use PyTorch only" -ForegroundColor White  
Write-Host "  [3] Try to run anyway" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1/2/3)"

switch ($choice) {
    "1" {
        Write-Host "`nInstalling tf-keras..." -ForegroundColor Yellow
        pip install tf-keras
        Write-Host "tf-keras installed!`n" -ForegroundColor Green
    }
    "2" {
        Write-Host "`nRemoving TensorFlow..." -ForegroundColor Yellow
        pip uninstall tensorflow tensorflow-intel -y
        Write-Host "Installing PyTorch (CPU)..." -ForegroundColor Yellow
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        Write-Host "Done!`n" -ForegroundColor Green
    }
    "3" {
        Write-Host "`nSkipping fix, attempting to run...`n" -ForegroundColor Yellow
    }
    default {
        Write-Host "`nInvalid choice, attempting to run anyway...`n" -ForegroundColor Yellow
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting SkillSync Application..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The app will be available at:" -ForegroundColor Green
Write-Host "  Local:   http://localhost:7860" -ForegroundColor White
Write-Host "  Network: http://0.0.0.0:7860" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the application
python app.py
