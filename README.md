# CYSNIC (Surveillance-Inspired UI Tracker)

CYSNIC is a high-performance, multi-target tracking system built for Apple Silicon (M-series). It features a dark-mode HUD skin mimicking surveillance and tactical aesthetics, built on a foundation of rigorous software engineering patterns.

## Prerequisites
Before compiling the project, you must have OpenCV (version 4.x or higher) installed on your system. 
This project is built and tested on macOS (Apple Silicon M-series).

The easiest way to install OpenCV on macOS is via Homebrew:

```bash
brew install opencv
```

**Note for Camera Access:** When running `cysnic_tracker 0` on macOS, your terminal (e.g. iTerm2 or Terminal.app) must be granted Camera permissions in System Settings -> Privacy & Security -> Camera.

## Building the Project
Once OpenCV is installed, CMake will handle downloading and compiling all other dependencies (Eigen3, Intel TBB, FFTW3, spdlog, GoogleTest) automatically.

```bash
mkdir build
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

## Running the Tracker
```bash
./cysnic_tracker /path/to/video.mp4
# Or for live camera:
./cysnic_tracker 0
```
