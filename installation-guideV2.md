# Installation Guide for Patient Positioning System

## Requirements
- Python 3.10 (MediaPipe not supported on Python 3.13)
- OpenCV-Contrib-Python
- MediaPipe
- NumPy

## Installation Steps

1. Download and install Python 3.10 from python.org (if not installed)

2. System installation:
```bash
# Run installation script
install.bat
```
This will:
- Create virtual environment
- Install all required packages
- Set up the project structure

3. Run the application:
```bash
# Run application with environment setup
setup_and_run.bat
```

## Directory Structure
```
patient_positioning/
├── setup.py
├── requirements.txt
├── install.bat
├── setup_and_run.bat
│
├── data/
│   └── patients/
│       ├── photos/        # Patient face photos
│       ├── features/      # Face recognition files
│       └── logs/         # Patient tracking log files
│
├── patient_positioning/
│   ├── main.py
│   ├── config/
│   ├── modules/
│   ├── utils/
│   └── tests/           
```

## Program Controls

### 1. Face Tracking with Recognition
- 's': Save reference position
- 'c': Clear visualization
- 'r': Reset tracking
- 'h': Hide/show points
- 'q': Return to menu

### 2. Body Pose Tracking
- Same controls as Face Tracking

### 3. Combined Tracking
Additional controls:
- 'f': Toggle face tracking
- 'p': Toggle pose tracking

### 4. Add New Patient
- Enter patient ID
- 's': Save face image
- 'q': Cancel

### 5. View Patient History
- View patient data and session history

## System Requirements
- Webcam or USB camera
- Minimum 4GB RAM
- Processor: Intel i3 or equivalent (i5 recommended)
- Operating System: Windows 10/11

## Support
For technical support or questions, please refer to the documentation in the docs/ directory.

## Installation Troubleshooting

### Common Issues:
1. Python version mismatch
   - Ensure Python 3.10 is installed
   - Check Python version in command prompt: `python --version`

2. Package installation failures
   - Run `pip install --upgrade pip`
   - Try installing packages individually if batch installation fails

3. Camera access issues
   - Check camera permissions in Windows settings
   - Ensure no other application is using the camera

4. Virtual environment problems
   - Delete venv folder and rerun install.bat
   - Check Python path in system environment variables

### Batch File Usage:
1. First time setup:
   - Run `install.bat`
   - Wait for all packages to install
   - Check for any error messages

2. Regular usage:
   - Run `setup_and_run.bat`
   - This will activate the environment and start the application

3. If errors occur:
   - Check console output for error messages
   - Try running installation again
   - Verify all requirements are met