rm -rf build
mkdir build

cmake -DCMAKE_BUILD_TYPE=Debug -DOptiX_INSTALL_DIR=$HOME/local/NVIDIA-OptiX-SDK-7.1.0-linux64-x86_64/ -Bbuild -H.

