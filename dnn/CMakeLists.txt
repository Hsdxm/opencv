set(DEMO    dnnDemo)

file(GLOB SRCS  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${DEMO}	${SRCS})
target_link_libraries(${DEMO} ${OpenCV_LIBS})
