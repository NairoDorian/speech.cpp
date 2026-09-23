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
    # transcribe.cpp ed3468f3 (#165): decode budget scales with audio length.
    add_transcribe_test(transcribe_decode_budget_unit tests/transcribe/decode_budget_unit.cpp)
    add_transcribe_test(transcribe_granite_diarize_parser_unit tests/transcribe/granite_diarize_parser_unit.cpp)
    add_transcribe_test(transcribe_log_unit tests/transcribe/log_unit.cpp)
    add_transcribe_test(transcribe_mel_unit tests/transcribe/mel_unit.cpp)
    add_transcribe_test(transcribe_moss_diarize_parser_unit tests/transcribe/moss_diarize_parser_unit.cpp)
    add_transcribe_test(transcribe_parakeet_chunked_limited_with_rc_mask_unit tests/transcribe/parakeet_chunked_limited_with_rc_mask_unit.cpp)
    add_transcribe_test(transcribe_parakeet_stream_ext_reject_unit tests/transcribe/parakeet_stream_ext_reject_unit.cpp)
    add_transcribe_test(transcribe_prefill_chunk_mask_unit tests/transcribe/prefill_chunk_mask_unit.cpp)
    add_transcribe_test(transcribe_run_dispatch_unit tests/transcribe/run_dispatch_unit.cpp)
    add_transcribe_test(transcribe_stream_capability_unit tests/transcribe/stream_capability_unit.cpp)
    add_transcribe_test(transcribe_stream_committed_pointer_stability tests/transcribe/stream_committed_pointer_stability.cpp)
    add_transcribe_test(transcribe_stream_dispatch_unit tests/transcribe/stream_dispatch_unit.cpp)
    add_transcribe_test(transcribe_teardown_safety_unit tests/transcribe/teardown_safety_unit.cpp)
    add_transcribe_test(transcribe_thread_default_unit tests/transcribe/thread_default_unit.cpp)
    add_transcribe_test(transcribe_tokenizer_decode_only_unit tests/transcribe/tokenizer_decode_only_unit.cpp)
    add_transcribe_test(transcribe_utf8_path_unit tests/transcribe/utf8_path_unit.cpp)

    # Whisper public-ABI contract gates (Phase 11 W2b). Ported from transcribe.cpp
    # in Phase 7.1 but never registered until 2026-09-23. They pin what the C ABI
    # promises for Whisper - language detection with no hint, SEGMENT timestamps
    # advertised and returned, the run ext (prompts, temperature ladder,
    # thresholds), chunk traces, abort, HF-identical BPE ids. They were the gate
    # the engine package had to pass before the arch retired (ledger B16c) and
    # now run against the engine through the ArchAdapter, so they need the
    # `whisper` model linked. Models come from scripts/fetch_asr_test_model.py;
    # each test exits 77 (SKIP) while its model is absent.
    set(_whisper_models "${CMAKE_CURRENT_SOURCE_DIR}/models")
    if ("whisper" IN_LIST AUDIOCPP_LINKED_MODELS)
        add_transcribe_test(transcribe_whisper_e2e_smoke tests/transcribe/whisper_e2e_smoke.cpp)
        add_transcribe_test(transcribe_whisper_tokenize_parity tests/transcribe/whisper_tokenize_parity.cpp)
        add_transcribe_test(transcribe_whisper_bin_e2e_smoke tests/transcribe/whisper_bin_e2e_smoke.cpp)
        add_transcribe_test(transcribe_whisper_bin_tokenize_parity tests/transcribe/whisper_bin_tokenize_parity.cpp)
        # The .bin parser / suppress-list units test the engine loader
        # (src/models/whisper/assets.cpp) since B16c deleted the arch's own
        # parser (src/runtime/transcribe-bin-loader.cpp).
        add_transcribe_test(transcribe_whisper_bin_parser_unit tests/transcribe/whisper_bin_parser_unit.cpp)
        add_transcribe_test(transcribe_whisper_bin_suppress_unit tests/transcribe/whisper_bin_suppress_unit.cpp)
    endif()
    if (ENGINE_BUILD_TESTS AND "whisper" IN_LIST AUDIOCPP_LINKED_MODELS)
        # The glossary-prompt check's thresholds (parent defaults: >= 5 hits,
        # margin > 3) are calibrated for a large model: the parent's own build
        # misses them with tiny, base and small (measured 2026-09-23). On the
        # pinned tiny the retired arch measured unprompted 0 -> prompted 2 in
        # both formats, so the gate holds tiny to exactly that; the engine
        # reproduces it.
        set(_whisper_tiny_glossary "TRANSCRIBE_WHISPER_GLOSSARY_MIN_HITS=2;TRANSCRIBE_WHISPER_GLOSSARY_MIN_MARGIN=1")
        set_tests_properties(transcribe_whisper_e2e_smoke PROPERTIES
            ENVIRONMENT "TRANSCRIBE_WHISPER_GGUF=${_whisper_models}/whisper-tiny-Q8_0.gguf;${_whisper_tiny_glossary}")
        set_tests_properties(transcribe_whisper_tokenize_parity PROPERTIES
            ENVIRONMENT "TRANSCRIBE_WHISPER_GGUF=${_whisper_models}/whisper-tiny-Q8_0.gguf")
        set_tests_properties(transcribe_whisper_bin_e2e_smoke PROPERTIES
            ENVIRONMENT "TRANSCRIBE_WHISPER_BIN_TINY_EN=${_whisper_models}/ggml-tiny.en.bin;TRANSCRIBE_WHISPER_BIN_TINY_Q8_0=${_whisper_models}/ggml-tiny.bin;${_whisper_tiny_glossary}")
        set_tests_properties(transcribe_whisper_bin_parser_unit PROPERTIES
            ENVIRONMENT "TRANSCRIBE_WHISPER_BIN_TINY_Q8_0=${_whisper_models}/ggml-tiny.bin")
        set_tests_properties(transcribe_whisper_bin_tokenize_parity PROPERTIES
            ENVIRONMENT "TRANSCRIBE_WHISPER_GGUF_TINY=${_whisper_models}/whisper-tiny-Q8_0.gguf;TRANSCRIBE_WHISPER_BIN_TINY_Q8_0=${_whisper_models}/ggml-tiny.bin")
    endif()

    # Phase 7 new unit tests
    add_transcribe_test(test_adapter_sniff_dispatch tests/unittests/test_adapter_sniff_dispatch.cpp)
    add_transcribe_test(test_adapter_run_params tests/unittests/test_adapter_run_params.cpp)

    # Engine-level tests link engine_runtime, never the engine_core OBJECT
    # library directly: engine_core is compiled with ENGINE_HAS_CUDA_ISTFT /
    # ENGINE_HAS_CUDA_TORCH_RANDOM under GGML_CUDA and calls kernels whose .cu
    # sources (and CUDA::cudart / CUDA::cufft) are attached to engine_runtime.
    # Linking engine_core alone works on CPU and fails to link on every CUDA
    # build (LNK2019 on CudaIstftRuntime, found 2026-09-23).
    add_executable(test_shared_weight_vram tests/unittests/test_shared_weight_vram.cpp)
    target_include_directories(test_shared_weight_vram PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_shared_weight_vram PRIVATE engine_runtime ggml ggml-cpu ggml-base cjson_vendor yaml_vendor sentencepiece)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME test_shared_weight_vram COMMAND test_shared_weight_vram)
    endif()

    # test_batch_dispatch links audiocpp, transcribe, engine_runtime
    # capi/include precedes include/ so "audiocpp.h" resolves to the Universal
    # C ABI header; upstream's include/audiocpp.h facade (opt-in) must not shadow it.
    add_executable(test_batch_dispatch tests/unittests/test_batch_dispatch.cpp)
    target_include_directories(test_batch_dispatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/capi/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe
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
        if ("moonshine" IN_LIST AUDIOCPP_LINKED_MODELS)
            add_test(NAME asr_e2e_edits_test
                     COMMAND asr_e2e_edits_test
                         "${CMAKE_CURRENT_SOURCE_DIR}/models/moonshine-tiny-Q8_0.gguf"
                         "${CMAKE_CURRENT_SOURCE_DIR}/assets/asr_validation/librispeech")
            set_tests_properties(asr_e2e_edits_test PROPERTIES SKIP_RETURN_CODE 2)
        endif()
    endif()

    # Phase 9 Parity and Contract Tests
    add_executable(frontend_contract_test tests/unittests/test_frontend_contract.cpp)
    target_include_directories(frontend_contract_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(frontend_contract_test PRIVATE engine_runtime ggml ggml-cpu ggml-base sentencepiece cjson_vendor yaml_vendor)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME frontend_contract_test COMMAND frontend_contract_test)
    endif()

    add_executable(frontend_parity_test tests/unittests/test_frontend_parity.cpp)
    target_include_directories(frontend_parity_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(frontend_parity_test PRIVATE engine_runtime ggml ggml-cpu ggml-base sentencepiece cjson_vendor yaml_vendor)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME frontend_parity_test COMMAND frontend_parity_test)
    endif()

    # Phase 8 contract tests. The tracker certified Phase 8 on StreamingSessionBase,
    # StreamChunker and RunControl, but these three sources were never compiled or
    # registered until 2026-09-23 (and, like the Phase 9 tests here, their bare
    # assert() checks vanished under Release's NDEBUG).
    foreach(_phase8_test streaming_session_base stream_chunker run_control)
        add_executable(${_phase8_test}_test tests/unittests/test_${_phase8_test}.cpp)
        target_include_directories(${_phase8_test}_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(${_phase8_test}_test PRIVATE engine_runtime ggml ggml-cpu ggml-base sentencepiece cjson_vendor yaml_vendor)
        if (ENGINE_BUILD_TESTS)
            add_test(NAME ${_phase8_test}_test COMMAND ${_phase8_test}_test)
        endif()
    endforeach()

    add_executable(tokenizer_parity_test tests/unittests/test_tokenizer_parity.cpp)
    target_include_directories(tokenizer_parity_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(tokenizer_parity_test PRIVATE engine_runtime ggml ggml-cpu ggml-base sentencepiece cjson_vendor yaml_vendor)
    if (ENGINE_BUILD_TESTS)
        add_test(NAME tokenizer_parity_test COMMAND tokenizer_parity_test)
    endif()
endif()
