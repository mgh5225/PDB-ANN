cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PDB-ANN)

include(FetchContent)

if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
	set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

FetchContent_Declare(cxxopts URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

list(APPEND CMAKE_PREFIX_PATH "~/Libs/libtorch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(lib)
file(GLOB_RECURSE LIBS "lib/*.cpp")
add_executable(main src/main.cpp ${LIBS})
target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}" cxxopts nlohmann_json::nlohmann_json)

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