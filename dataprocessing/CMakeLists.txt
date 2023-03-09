project("DeepSDF")
cmake_minimum_required(VERSION 3.0)

find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nanoflann REQUIRED)
find_package(Pangolin REQUIRED)

add_subdirectory(third-party/cnpy)

set(SOURCES
    ShaderProgram.cpp
    Utils.cpp)

add_executable(PreprocessMesh PreprocessMesh.cpp ${SOURCES})
target_include_directories(PreprocessMesh PRIVATE third-party/cnpy)
target_link_libraries(PreprocessMesh PRIVATE CLI11::CLI11 pangolin cnpy Eigen3::Eigen nanoflann::nanoflann)
target_compile_features(PreprocessMesh PRIVATE cxx_std_17)
set_target_properties(PreprocessMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(SampleVisibleMeshSurface SampleVisibleMeshSurface.cpp ${SOURCES})
target_include_directories(SampleVisibleMeshSurface PRIVATE third-party/cnpy)
target_link_libraries(SampleVisibleMeshSurface PRIVATE CLI11::CLI11 pangolin cnpy Eigen3::Eigen nanoflann::nanoflann ${Pangolin_LIBRARIES})
target_compile_features(SampleVisibleMeshSurface PRIVATE cxx_std_17)
set_target_properties(SampleVisibleMeshSurface PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
