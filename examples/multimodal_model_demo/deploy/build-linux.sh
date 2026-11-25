set -e
rm -rf build
mkdir build && cd build

GCC_COMPILER=/home/top/CodeField/Qwen3.5/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu
cmake .. -DCMAKE_CXX_COMPILER=${GCC_COMPILER}/bin/aarch64-none-linux-gnu-g++  \
        -DCMAKE_C_COMPILER=${GCC_COMPILER}/bin/aarch64-none-linux-gnu-gcc \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \

make -j8
make install
