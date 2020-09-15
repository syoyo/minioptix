# Mini OptiX sample

Minimal dependency OptiX sample.

* Use CUDA Driver API only
  * Use cuew(no CUDA SDK required)

## Requirements

* C++11 or later
* CMake
* (optional) ninja build
* OptiX 7.1(OptiX 6 would work) SDK(header files)

## Supported platforms

* 64bit Linux x86-64(e.g. Ubuntu 18.04)
* Windows 10
  * Visual Studio 2019

## Compile a shader

```
cd <optix_sdk>/SDK/optixTriangle
nvcc --ptx -I../../include -I../ optixTriangle.cu

# copy optixTriangle.ptx to <minioptix>/data/
```

## Thrid party licenses

* SDL2. zlib license: https://www.libsdl.org/index.php
* imgui. MIT license. https://github.com/ocornut/imgui
* imgui_sdl. MIT license.
* stb_image_write: public domain(or MIT licensed). https://github.com/nothings/stb
