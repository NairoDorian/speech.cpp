# lint_teardown.cmake - forbid raw ggml teardown calls in library code.
#
# Run as: cmake -DSRC_DIR=<repo>/src -P lint_teardown.cmake
#
# Keep library code on the no-throw wrappers. Raw ggml backend teardown calls
# are allowed only inside the safe-teardown helper unit (transcribe-backend.{h,cpp}).
#
# NOTE (speech.cpp): this gate is currently scoped to src/runtime/ (the transcribe
# runtime) where the safe_* wrappers from transcribe-backend.h are already in use.
# The broader src/ tree still contains raw ggml_backend_free/buffer_free/sched_free
# calls inherited from the audio.cpp model sessions — converting those is the
# Phase 0 sub-task 0.J "safe_* teardown retrofit" (file-by-file migration gated
# by this same script). When that retrofit lands, expand SRC_DIR here to ${CMAKE_SOURCE_DIR}/src
# to cover the full tree.

if(NOT DEFINED SRC_DIR)
    message(FATAL_ERROR "pass -DSRC_DIR=<repo>/src")
endif()

file(GLOB_RECURSE lint_files "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.h")

set(violations "")
foreach(f IN LISTS lint_files)
    if(f MATCHES "transcribe-backend\\.(cpp|h)$")
        continue()
    endif()
    file(READ "${f}" content)
    foreach(fn ggml_backend_free ggml_backend_buffer_free ggml_backend_sched_free)
        if(content MATCHES "${fn}[ \t]*\\(")
            string(APPEND violations "  ${f}: ${fn}(\n")
        endif()
    endforeach()
endforeach()

if(violations)
    message(FATAL_ERROR
        "raw ggml teardown calls in library code — use the no-throw "
        "wrappers transcribe::safe_backend_free / safe_buffer_free / "
        "safe_sched_free from transcribe-backend.h instead:\n${violations}")
endif()

message(STATUS "lint_teardown: no raw ggml teardown calls in ${SRC_DIR}")
