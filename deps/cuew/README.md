syoyo's modification to CUEW.

Currently tests on Ubuntu 18.04.

## Changes

* Support CUDA 10.2
* Support CUDNN 7.6

## Supported API

* [x] cuda.h
* [x] cudnn.h
* [x] nvrtc.h
* [ ] cudaGL.h

## Generate cuew

You need python3.
Install pycparser.

```
$ python -m pip install pycparser
```

```
$ cd auto
# Edit header path in cuew_ge.py if required
$ ./cuew_gen.sh
```

## Buld tests

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Known issues

* Combining with Address Sanitizer(`-fsanitizer=address`) won't work(calling CUDA API results in undefined behavior)
* CUEW does not warn when using deprecated API

## TODO

* [ ] Test on MSVC.
* [ ] Support cudaGL.h
* [x] Keep up with CUDA 10.2
* [ ] Test cuDNN API call.
* [ ] Find a way to co-exist with Address Sanitizer

=================

The CUDA Extension Wrangler Library (CUEW) is a cross-platform open-source
C/C++ extension loading library. CUEW provides efficient run-time mechanisms
for determining which CUDA functions and extensions extensions are supported
on the target platform.

CUDA core and extension functionality is exposed in a single header file.
CUEW has been tested on a variety of operating systems, including Windows,
Linux, Mac OS X.

LICENSE

CUEW library is released under the Apache 2.0 license.

