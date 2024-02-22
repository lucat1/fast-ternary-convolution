# Source files
set(asl_src
  fib.cc
)

# Compile the library
add_library(${ASL_LIB_NAME} ${ASL_LIB_TYPE} ${asl_src})

# How and what to install
install(TARGETS ${ASL_LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ../include/asl DESTINATION include)
