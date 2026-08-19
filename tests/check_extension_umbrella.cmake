if(NOT DEFINED SPEECHCPP_SOURCE_DIR)
    message(FATAL_ERROR "SPEECHCPP_SOURCE_DIR is required")
endif()

set(include_dir "${SPEECHCPP_SOURCE_DIR}/include/transcribe")
set(umbrella "${include_dir}/extensions.h")

if(NOT EXISTS "${umbrella}")
    message(FATAL_ERROR "missing umbrella header: ${umbrella}")
endif()

file(READ "${umbrella}" umbrella_contents)
file(GLOB family_headers "${include_dir}/*.h")

set(missing)
foreach(header IN LISTS family_headers)
    get_filename_component(name "${header}" NAME)
    # Skip extensions.h itself (the umbrella file).
    # Skip transcribe.h — it is the base ABI header, included directly as
    # `#include "transcribe.h"` (not `#include "transcribe/transcribe.h"`)
    # because the public include path is include/transcribe/, not include/.
    if(name STREQUAL "extensions.h" OR name STREQUAL "transcribe.h")
        continue()
    endif()

    set(expected "#include \"transcribe/${name}\"")
    string(FIND "${umbrella_contents}" "${expected}" found_at)
    if(found_at EQUAL -1)
        list(APPEND missing "${expected}")
    endif()
endforeach()

if(missing)
    string(REPLACE ";" "\n  " missing_lines "${missing}")
    message(FATAL_ERROR
        "include/transcribe/extensions.h is missing family header includes:\n"
        "  ${missing_lines}")
endif()
