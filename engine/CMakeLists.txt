cmake_minimum_required(VERSION 3.14)
project(PetRWKVEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(ENABLE_NCNN_BACKEND "Enable ncnn backend" OFF)
option(ENABLE_QNN_BACKEND "Enable QNN backend" OFF)
option(ENABLE_TESTING "Enable tests" OFF)

set(ENGINE_SRC
    src/rwkv_engine.cpp
    src/rwkv_v7.cpp
    src/ops/rwkv_math.h
    src/rwkv_v7.h
    src/weight_loader.cpp
    src/weight_loader.h
    src/tokenizer.cpp
    src/tokenizer.h
)

set(ENGINE_HEADERS
    src/rwkv_engine.h
    src/rwkv_v7.h
    src/ops/rwkv_math.h
    src/weight_loader.h
    src/tokenizer.h
)

add_library(${PROJECT_NAME} ${ENGINE_SRC})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -ffast-math -fomit-frame-pointer
    $<$<CONFIG:Debug>:-g -O0 -DDEBUG>
)

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

if(ANDROID)
    target_compile_options(${PROJECT_NAME} PRIVATE -fPIC -D__ANDROID__)
    if(${ANDROID_ABI} STREQUAL "arm64-v8a")
        target_compile_options(${PROJECT_NAME} PRIVATE -march=armv8.2a+fp16+dotprod)
    elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
        target_compile_options(${PROJECT_NAME} PRIVATE -march=armv7-a+fp -mfpu=neon -mfloat-abi=softfp)
    endif()
endif()

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES ${ENGINE_HEADERS} DESTINATION include/petrwkv)

if(ENABLE_TESTING)
    enable_testing()
    add_executable(petrwkv_test tests/test_engine.cpp)
    target_link_libraries(petrwkv_test PRIVATE ${PROJECT_NAME})
    add_test(NAME engine_test COMMAND petrwkv_test)
endif()