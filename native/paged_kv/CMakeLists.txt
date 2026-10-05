cmake_minimum_required(VERSION 3.24)
project(mlx2_paged_kv_native LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(MLX REQUIRED CONFIG)
get_target_property(_mlx_imported_path mlx IMPORTED_LOCATION_RELEASE)
if(NOT _mlx_imported_path MATCHES "[/]site-packages[/]mlx[/]lib[/]libmlx[.]dylib$")
  message(FATAL_ERROR "Build must use the pinned wheel's libmlx.dylib: ${_mlx_imported_path}")
endif()
if(NOT MLX_LIBRARY STREQUAL _mlx_imported_path)
  message(FATAL_ERROR "MLX_LIBRARY must resolve to the pinned wheel: ${MLX_LIBRARY}")
endif()

include(FetchContent)
FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG a47983000e9dda68177fc1fd426f7147d785c2c0
  GIT_SHALLOW FALSE)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(_paged_kv_native NB_STATIC STABLE_ABI NB_DOMAIN mlx binding.cpp arena.cpp)
target_compile_features(_paged_kv_native PRIVATE cxx_std_20)
target_compile_options(_paged_kv_native PRIVATE -fblocks -Wall -Wextra -Werror)
target_link_libraries(_paged_kv_native PRIVATE mlx Python::Module)
set_target_properties(_paged_kv_native PROPERTIES PREFIX "" BUILD_RPATH "${MLX_LIBRARY_DIRS}")
install(TARGETS _paged_kv_native LIBRARY DESTINATION .)
