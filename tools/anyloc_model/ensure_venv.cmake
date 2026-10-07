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

# Creates the venv the AnyLoc ONNX export runs in, from scratch, and installs the pinned requirements into it.
# Run in script mode by the anyloc_venv build step:
#   cmake -DPYTHON=<interpreter> -DVENV=<dir> -DREQUIREMENTS=<requirements.txt> -DSTAMP=<file> -P ensure_venv.cmake
# STAMP is written last and lives inside VENV, so a venv that is deleted or only half built is built again.

foreach(var PYTHON VENV REQUIREMENTS STAMP)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "ensure_venv.cmake needs -D${var}=...")
    endif()
endforeach()

# Starting over rather than upgrading in place is what keeps the venv equal to the requirements file: pip does not
# remove packages that a previous version of the file pulled in.
file(REMOVE_RECURSE "${VENV}")

execute_process(COMMAND "${PYTHON}" -m venv "${VENV}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Could not create the AnyLoc export venv in ${VENV} with ${PYTHON} (needs the venv module, "
                        "e.g. the python3-venv package)")
endif()

execute_process(
    COMMAND "${VENV}/bin/python" -m pip install --disable-pip-version-check --quiet -r "${REQUIREMENTS}"
    RESULT_VARIABLE result
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Could not install ${REQUIREMENTS} into ${VENV}. It needs access to pypi.org and "
                        "download.pytorch.org; on a machine without it, set CUVSLAM_ANYLOC_PYTHON to an interpreter "
                        "that already has these packages.")
endif()

file(SHA256 "${REQUIREMENTS}" requirements_hash)
file(WRITE "${STAMP}" "${requirements_hash}\n")
