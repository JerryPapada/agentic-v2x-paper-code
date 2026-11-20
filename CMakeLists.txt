# LLM rApp-assisted V2X scheduler scenarios.

macro(llm_v2x_build_variant VARIANT_NAME SRC_DIR OUT_DIR)
    file(GLOB _src CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_DIR}/*.cc)
    if(_src)
        build_exec(
            EXECNAME ${VARIANT_NAME}
            EXECNAME_PREFIX scratch_llm_v2x_scheduler_
            SOURCE_FILES "${_src}"
            LIBRARIES_TO_LINK "${ns3-libs}" "${ns3-contrib-libs}"
            EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/llm_v2x_scheduler/${OUT_DIR}/
        )
    endif()
endmacro()

llm_v2x_build_variant(scenario_llm_v2x_s20 ns3/s20 ns3/s20)
llm_v2x_build_variant(scenario_llm_v2x_s25 ns3/s25 ns3/s25)
llm_v2x_build_variant(scenario_llm_v2x_s30 ns3/s30 ns3/s30)

