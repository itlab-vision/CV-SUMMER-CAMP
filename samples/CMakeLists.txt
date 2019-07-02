# Get all cpp files in current directory.
file(GLOB app_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(app_filename ${app_list})
  # Get file name without extension.
  get_filename_component(app ${app_filename} NAME_WE)
  # Add and configure executable file to be produced.
  add_executable(${app} ${app_filename})
  if (UNIX)
      target_link_libraries(${target} ${app} ${CMAKE_THREAD_LIBS_INIT})
  endif (UNIX)
  target_link_libraries(${target} ${app} ${LIBRARY_DEPS} )
endforeach()