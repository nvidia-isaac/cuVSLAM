# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

# Third-party notices of the binary packages, assembled from the license files of the fetched sources.
#
# Every cmake/ext file calls one of
#   cuvslam_third_party_notice()          for code or data that ends up in the C++ package or the wheels
#   cuvslam_third_party_not_distributed() for anything else (tests, examples, development-only options)
# next to its version pin. cuvslam_check_third_party_coverage() fails the configure for a file that calls neither.
#
# Script mode compares a generated file with the checked-in one:
#   cmake -DGENERATED=<file> -DCHECKED_IN=<file> -DUPDATE_COMMAND=<text> -P ThirdPartyNotices.cmake

# cuvslam_third_party_notice(NAME <name> VERSION <version> URL <url> LICENSE <SPDX expression>
#                            [FILES <license or NOTICE file>...] [TEXT <license text>] [NOTE <text>])
function(cuvslam_third_party_notice)
    cmake_parse_arguments(ARG "" "NAME;VERSION;URL;LICENSE;TEXT;NOTE" "FILES" ${ARGN})
    foreach(_required NAME VERSION URL LICENSE)
        if(NOT ARG_${_required})
            message(FATAL_ERROR "cuvslam_third_party_notice(${ARG_NAME}): ${_required} is required")
        endif()
    endforeach()
    if(NOT ARG_FILES AND NOT ARG_TEXT)
        message(FATAL_ERROR "cuvslam_third_party_notice(${ARG_NAME}): give the license as FILES or TEXT")
    endif()

    string(REPEAT "=" 80 _rule)
    set(_section "${_rule}\n${ARG_NAME}\n${_rule}\n\n")
    string(APPEND _section "Version: ${ARG_VERSION}\nSource:  ${ARG_URL}\nLicense: ${ARG_LICENSE}\n")
    if(ARG_NOTE)
        string(APPEND _section "\n${ARG_NOTE}\n")
    endif()
    foreach(_file IN LISTS ARG_FILES)
        if(NOT EXISTS "${_file}")
            message(FATAL_ERROR "cuvslam_third_party_notice(${ARG_NAME}): ${_file} does not exist")
        endif()
        file(READ "${_file}" _text)
        string(REGEX REPLACE "^([ \t\r]*\n)+" "" _text "${_text}")
        string(REGEX REPLACE "(\n[ \t\r]*)+$" "" _text "${_text}")
        get_filename_component(_file_name "${_file}" NAME)
        string(APPEND _section "\n--- ${_file_name} ---\n\n${_text}\n")
    endforeach()
    if(ARG_TEXT)
        string(APPEND _section "\n${ARG_TEXT}\n")
    endif()

    string(MAKE_C_IDENTIFIER "${ARG_NAME}" _key)
    set_property(GLOBAL APPEND PROPERTY CUVSLAM_THIRD_PARTY_NOTICE_KEYS "${_key}")
    set_property(GLOBAL PROPERTY "CUVSLAM_THIRD_PARTY_NOTICE_${_key}" "${_section}")
endfunction()

# cuvslam_third_party_not_distributed(<name> <reason>): no package ships <name>; the call records why.
function(cuvslam_third_party_not_distributed name reason)
endfunction()

# Every cmake/ext file has to decide whether its dependency ships.
function(cuvslam_check_third_party_coverage ext_dir)
    file(GLOB _ext_files "${ext_dir}/*.cmake")
    foreach(_file IN LISTS _ext_files)
        file(READ "${_file}" _text)
        if(NOT _text MATCHES "cuvslam_third_party_(notice|not_distributed)\\(")
            message(FATAL_ERROR "${_file} declares neither cuvslam_third_party_notice() nor "
                                "cuvslam_third_party_not_distributed(). Call the first if its code or data ends up "
                                "in a binary package, the second otherwise; see cmake/ThirdPartyNotices.cmake.")
        endif()
    endforeach()
endfunction()

# cuvslam_write_third_party_notices(<output> [BASE <notices file to extend>])
function(cuvslam_write_third_party_notices output)
    cmake_parse_arguments(ARG "" "BASE" "" ${ARGN})
    if(ARG_BASE)
        file(READ "${ARG_BASE}" _content)
    else()
        set(_content "cuVSLAM third-party notices\n\n")
        string(APPEND _content
            "The cuVSLAM binary packages, the C++ package and the Python wheels, contain the third-party\n"
            "software listed below, each under its own license. cuVSLAM itself is licensed under the NVIDIA\n"
            "Community License, see LICENSE.\n\n"
            "CMake generates this file from the license files of the sources it fetches\n"
            "(cmake/ThirdPartyNotices.cmake). The copy in the Python wheels also lists nanobind and\n"
            "tsl::robin_map, which are compiled into the Python extension.\n")
    endif()

    get_property(_keys GLOBAL PROPERTY CUVSLAM_THIRD_PARTY_NOTICE_KEYS)
    if(_keys)
        list(REMOVE_DUPLICATES _keys)
        list(SORT _keys CASE INSENSITIVE)
    endif()
    foreach(_key IN LISTS _keys)
        get_property(_section GLOBAL PROPERTY "CUVSLAM_THIRD_PARTY_NOTICE_${_key}")
        string(APPEND _content "\n${_section}")
    endforeach()

    # As pre-commit's whitespace and end-of-file fixers would leave it, so the checked-in copy stays identical
    string(REGEX REPLACE "[ \t\r]+\n" "\n" _content "${_content}")
    string(REGEX REPLACE "\n+$" "\n" _content "${_content}")
    file(WRITE "${output}.tmp" "${_content}")
    configure_file("${output}.tmp" "${output}" COPYONLY)
    file(REMOVE "${output}.tmp")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    file(SHA256 "${GENERATED}" _generated_hash)
    file(SHA256 "${CHECKED_IN}" _checked_in_hash)
    if(NOT _generated_hash STREQUAL _checked_in_hash)
        message(FATAL_ERROR "${CHECKED_IN} does not match the notices this build generated, ${GENERATED}: "
                            "a dependency was added, removed or changed. Update it with\n  ${UPDATE_COMMAND}")
    endif()
endif()
