cmake_minimum_required(VERSION 3.19)

project(VolumeRender CXX)

set(PROJECT_CONTENT "${CMAKE_CURRENT_SOURCE_DIR}/content")
set(PROJECT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_OUTPUT_DIRECTORY "${PROJECT_DIRECTORY}/x64")
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_DIRECTORY}/cmake")

# # 检测编译器并设置相应的标志
# if(MSVC)
# # 使用静态运行时库
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
# else()
# # 其他编译器的设置（如果需要）
# endif()
include(cmake_settings)
include(compiler_settings)

include(3rd-party/fmt)
include(3rd-party/glfw)
include(3rd-party/imgui)
include(3rd-party/implot)
include(3rd-party/directx-tex)
include(3rd-party/nlohmann)

find_package(OpenCV REQUIRED)

set(INCLUDE
    include/Application.h
    include/ApplicationVolumeRender.h
    include/Common.h
    include/TransferFunction.h
)

set(SOURCE
    source/Application.cpp
    source/ApplicationVolumeRender.cpp
    source/Main.cpp
)

file(GLOB SHADERS "content/Shaders/*.hlsl")

source_group("include" FILES ${INCLUDE})
source_group("source" FILES ${SOURCE})
source_group("shaders" FILES ${SHADERS})

add_executable(VolumeRender ${INCLUDE} ${SOURCE} ${SHADERS})

target_link_libraries(VolumeRender PRIVATE fmt glfw imgui implot nlohmann_json directx-tex d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib)
target_link_libraries(VolumeRender PRIVATE ${OpenCV_LIBS})

target_include_directories(VolumeRender PRIVATE "include")

set_target_properties(VolumeRender PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIRECTORY}")
set_source_files_properties(${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

set(PROJECT_CONTENT "${CMAKE_CURRENT_SOURCE_DIR}/content")

# 将content目录复制到输出目录
add_custom_command(TARGET VolumeRender POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_CONTENT}
    $<TARGET_FILE_DIR:VolumeRender>/content
)