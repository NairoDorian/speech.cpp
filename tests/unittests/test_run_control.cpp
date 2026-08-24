#include "engine/framework/runtime/run_control.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace engine::runtime;

static void test_abort_polling() {
    RunControl rc;
    assert(!rc.poll_abort());

    rc.request_abort();
    assert(rc.poll_abort());

    rc.reset_abort();
    assert(!rc.poll_abort());
    std::cout << "[PASS] test_abort_polling" << std::endl;
}

static void test_progress_emission() {
    RunControl rc;
    std::vector<ProgressInfo> recorded;

    rc.set_progress_callback([&recorded](const ProgressInfo & info) {
        recorded.push_back(info);
        return true;
    });

    rc.emit_progress("asr_stage", 5, 10);
    assert(recorded.size() == 1);
    assert(recorded[0].stage == "asr_stage");
    assert(recorded[0].completed_units == 5);
    assert(recorded[0].total_units == 10);
    assert(recorded[0].progress == 0.5f);

    // Cancel requested from callback
    rc.set_progress_callback([](const ProgressInfo &) {
        return false;
    });

    bool caught = false;
    try {
        rc.emit_progress("asr_stage", 6, 10);
    } catch (const ProgressCanceled & e) {
        caught = true;
    }
    assert(caught);

    std::cout << "[PASS] test_progress_emission" << std::endl;
}

static void test_abort_triggers_progress_canceled() {
    RunControl rc;
    bool called = false;
    rc.set_progress_callback([&called](const ProgressInfo &) {
        called = true;
        return true;
    });

    rc.request_abort();
    bool caught = false;
    try {
        rc.emit_progress("test", 1, 1);
    } catch (const ProgressCanceled &) {
        caught = true;
    }
    assert(caught);
    assert(!called);  // Callback must not be invoked if already aborted

    std::cout << "[PASS] test_abort_triggers_progress_canceled" << std::endl;
}

int main() {
    test_abort_polling();
    test_progress_emission();
    test_abort_triggers_progress_canceled();
    std::cout << "All RunControl unit tests passed successfully." << std::endl;
    return 0;
}
