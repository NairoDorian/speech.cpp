# cmake/transcribe-tests.cmake — Test registration for ported transcribe unit tests & Phase 7 gates

if (SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES)
    add_library(transcribe_wav_helper STATIC
        tests/transcribe/wav.cpp
    )
    target_include_directories(transcribe_wav_helper PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcribe
    )

    function(add_transcribe_test test_name source_file)
        add_executable(${test_name} ${source_file})
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcribe
        )
        target_compile_definitions(${test_name} PRIVATE
            TRANSCRIBE_BUILD
            "TRANSCRIBE_TEST_FIXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures\""
            "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/samples\""
        )
        target_link_libraries(${test_name} PRIVATE transcribe_internal transcribe_wav_helper ggml ggml-cpu ggml-base)
        if (ENGINE_BUILD_TESTS)
            add_test(NAME ${test_name} COMMAND ${test_name})
            set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endfunction()

    # Pure C ABI smoke
    add_executable(transcribe_api_smoke tests/transcribe/api_smoke.c)
    set_target_properties(transcribe_api_smoke PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        LINKER_LANGUAGE C
    )
    target_include_directories(transcribe_api_smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe
    )
    target_link_libraries(transcribe_api_smoke PRIVATE transcribe)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME transcribe_api_smoke COMMAND transcribe_api_smoke)
    endif()

    # Ported transcribe unit tests (Batch 1: Core plumbing & unit tests)
    add_transcribe_test(transcribe_backend_classification_unit tests/transcribe/backend_classification_unit.cpp)
    add_transcribe_test(transcribe_backend_init_throw_unit tests/transcribe/backend_init_throw_unit.cpp)
    add_transcribe_test(transcribe_backend_init_unit tests/transcribe/backend_init_unit.cpp)
    add_transcribe_test(transcribe_backend_metal_simdgroup_gate_unit tests/transcribe/backend_metal_simdgroup_gate_unit.cpp)
    add_transcribe_test(transcribe_backend_probe_order_unit tests/transcribe/backend_probe_order_unit.cpp)
    add_transcribe_test(transcribe_batch_mask_unit tests/transcribe/batch_mask_unit.cpp)
    add_transcribe_test(transcribe_conv_pw_promote_unit tests/transcribe/conv_pw_promote_unit.cpp)
    add_transcribe_test(transcribe_debug_dump_unit tests/transcribe/debug_dump_unit.cpp)
    add_transcribe_test(transcribe_granite_diarize_parser_unit tests/transcribe/granite_diarize_parser_unit.cpp)
    add_transcribe_test(transcribe_log_unit tests/transcribe/log_unit.cpp)
    add_transcribe_test(transcribe_mel_unit tests/transcribe/mel_unit.cpp)
    add_transcribe_test(transcribe_moss_diarize_parser_unit tests/transcribe/moss_diarize_parser_unit.cpp)
    add_transcribe_test(transcribe_parakeet_chunked_limited_with_rc_mask_unit tests/transcribe/parakeet_chunked_limited_with_rc_mask_unit.cpp)
    add_transcribe_test(transcribe_parakeet_stream_ext_reject_unit tests/transcribe/parakeet_stream_ext_reject_unit.cpp)
    add_transcribe_test(transcribe_prefill_chunk_mask_unit tests/transcribe/prefill_chunk_mask_unit.cpp)
    add_transcribe_test(transcribe_run_dispatch_unit tests/transcribe/run_dispatch_unit.cpp)
    add_transcribe_test(transcribe_sortformer_stream_ext_unit tests/transcribe/sortformer_stream_ext_unit.cpp)
    add_transcribe_test(transcribe_stream_capability_unit tests/transcribe/stream_capability_unit.cpp)
    add_transcribe_test(transcribe_stream_committed_pointer_stability tests/transcribe/stream_committed_pointer_stability.cpp)
    add_transcribe_test(transcribe_stream_dispatch_unit tests/transcribe/stream_dispatch_unit.cpp)
    add_transcribe_test(transcribe_teardown_safety_unit tests/transcribe/teardown_safety_unit.cpp)
    add_transcribe_test(transcribe_thread_default_unit tests/transcribe/thread_default_unit.cpp)
    add_transcribe_test(transcribe_tokenizer_decode_only_unit tests/transcribe/tokenizer_decode_only_unit.cpp)
    add_transcribe_test(transcribe_utf8_path_unit tests/transcribe/utf8_path_unit.cpp)
    add_transcribe_test(transcribe_whisper_bin_parser_unit tests/transcribe/whisper_bin_parser_unit.cpp)
    add_transcribe_test(transcribe_whisper_bin_suppress_unit tests/transcribe/whisper_bin_suppress_unit.cpp)

    # Phase 7 new unit tests
    add_transcribe_test(test_adapter_sniff_dispatch tests/unittests/test_adapter_sniff_dispatch.cpp)

    # test_shared_weight_vram links engine_core and ggml
    add_executable(test_shared_weight_vram tests/unittests/test_shared_weight_vram.cpp)
    target_include_directories(test_shared_weight_vram PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_shared_weight_vram PRIVATE engine_core ggml ggml-cpu ggml-base cjson_vendor yaml_vendor sentencepiece)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME test_shared_weight_vram COMMAND test_shared_weight_vram)
    endif()

    # test_batch_dispatch links audiocpp, transcribe, engine_runtime
    add_executable(test_batch_dispatch tests/unittests/test_batch_dispatch.cpp)
    target_include_directories(test_batch_dispatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe
        ${CMAKE_CURRENT_SOURCE_DIR}/capi/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
    )
    target_compile_definitions(test_batch_dispatch PRIVATE TRANSCRIBE_BUILD)
    target_link_libraries(test_batch_dispatch PRIVATE audiocpp transcribe_internal engine_runtime cjson_vendor yaml_vendor sentencepiece ggml ggml-cpu ggml-base)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME test_batch_dispatch COMMAND test_batch_dispatch)
    endif()

    # test_family_registry links engine_runtime
    add_executable(test_family_registry tests/unittests/test_family_registry.cpp)
    target_include_directories(test_family_registry PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_family_registry PRIVATE engine_runtime cjson_vendor yaml_vendor sentencepiece ggml ggml-cpu ggml-base)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME family_registry_unit COMMAND test_family_registry)
    endif()

    # asr_e2e_edits_test
    add_executable(asr_e2e_edits_test tests/asr_e2e_edits_test.cpp)
    target_include_directories(asr_e2e_edits_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(asr_e2e_edits_test PRIVATE transcribe)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME asr_e2e_edits_test
                 COMMAND asr_e2e_edits_test
                     "${CMAKE_CURRENT_SOURCE_DIR}/models/moonshine-tiny-Q8_0.gguf"
                     "${CMAKE_CURRENT_SOURCE_DIR}/assets/asr_validation/librispeech")
        set_tests_properties(asr_e2e_edits_test PROPERTIES SKIP_RETURN_CODE 2)
    endif()

    # Phase 9 Parity and Contract Tests
    add_executable(frontend_contract_test tests/unittests/test_frontend_contract.cpp)
    target_include_directories(frontend_contract_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(frontend_contract_test PRIVATE engine_core ggml ggml-cpu ggml-base sentencepiece cjson_vendor yaml_vendor)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME frontend_contract_test COMMAND frontend_contract_test)
    endif()

    add_executable(frontend_parity_test tests/unittests/test_frontend_parity.cpp)
    target_include_directories(frontend_parity_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(frontend_parity_test PRIVATE engine_core ggml ggml-cpu ggml-base sentencepiece cjson_vendor yaml_vendor)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME frontend_parity_test COMMAND frontend_parity_test)
    endif()

    add_executable(tokenizer_parity_test tests/unittests/test_tokenizer_parity.cpp)
    target_include_directories(tokenizer_parity_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(tokenizer_parity_test PRIVATE engine_core ggml ggml-cpu ggml-base sentencepiece cjson_vendor yaml_vendor)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME tokenizer_parity_test COMMAND tokenizer_parity_test)
    endif()
endif()
