cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PDB-ANN)

list(APPEND CMAKE_PREFIX_PATH "~/Libs/libtorch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(lib)
file(GLOB_RECURSE LIBS "lib/*.cpp")
add_executable(main src/main.cpp ${LIBS})
target_link_libraries(main "${TORCH_LIBRARIES}")
target_include_directories(main PRIVATE "~/Libs/json/single_include")
set_property(TARGET main PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)