# Mini OptiX sample

Minimal dependency OptiX sample.

* Use CUDA Driver API only
  * Use cuew(no CUDA SDK required)

## Requirements

* C++11 or later
* CMake
* (optional) ninja build
* OptiX 7.1(OptiX 6 would work) SDK(header files)

## Compile a shader

```
cd <optix_sdk>/SDK/optixHello
nvcc --ptx -I../../include -I../ draw_solid_color.cu

# copy draw_solid_color.ptx to <minioptix>/data/
```

## Thrid party licenses

* SDL2. zlib license: https://www.libsdl.org/index.php
* imgui. MIT license. https://github.com/ocornut/imgui
* imgui_sdl. MIT license.
* stb_image_write: public domain(or MIT licensed). https://github.com/nothings/stb
