# CYSNIC Tactical Tracker

CYSNIC is a high-performance, multi-target tracking system built for Apple Silicon (M-series).

## Prerequisites
Before compiling the project, you must have OpenCV installed on your system.
The easiest way to install OpenCV on macOS is via Homebrew:

```bash
brew install opencv
```

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
