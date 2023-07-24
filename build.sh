#!/bin/bash

LIBTORCH=/home/patrick/Libs/libtorch

[ ! -d ./build ] && mkdir build

cd build

cmake -DCMAKE_PREFIX_PATH=$LIBTORCH ..
cmake --build . --config Release
