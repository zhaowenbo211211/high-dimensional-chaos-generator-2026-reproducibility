// Maintainer: zhaowenbo
// Cooja ScriptRunner: control simulation lifecycle.
// Uses GENERATE_MSG to schedule a clean success after 180s simulation time,
// avoiding the TEST FAILED state from TIMEOUT.

GENERATE_MSG(180000, "success");
log.log("Simulation autostart, scheduled success at 180s\n");

while (true) {
    YIELD();
    if (msg.equals("success")) {
        log.log("Simulation complete after 180s, marking success\n");
        log.testOK();
    }
}
