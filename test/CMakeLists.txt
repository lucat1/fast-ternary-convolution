FetchContent_Declare(
  googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           f8d7d77c06936315286eb55f8de22cd23c188571 # version 1.14.0
)

FetchContent_MakeAvailable(
  googletest
)

set(test_src
  run_all.cc
  fib_test.cc
)

add_executable(test_asl ${test_src})

target_link_libraries(test_asl
  gtest
  ${ASL_LIB_NAME}
)

add_test(
  NAME tests
  COMMAND test_asl
)
