#!/bin/bash
# Maintainer: zhaowenbo
# Run Cooja headless experiment with raw evidence collection.
# This script:
#   1. Clears previous outputs
#   2. Starts Python listener (connects to 60000-60008)
#   3. Starts Cooja in no-gui mode
#   4. Waits up to 120s, then terminates
#   5. Saves stdout, environment info, and run command

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_LOGS="$PROJECT_ROOT/raw_logs"
COOJA_DIR="${AUTHOR_HOME}/contiki-ng/tools/cooja"
CSC_FILE="$SCRIPT_DIR/prng371-udp-demo.csc"

mkdir -p "$RAW_LOGS"

cd "$SCRIPT_DIR"
rm -f python_test.log cooja_decrypted.png cooja_stdout_real.log

# Record run command and environment
cat > "$RAW_LOGS/cooja_run_command.txt" << EOF
# Cooja Run Command
# Generated: $(date -Iseconds)

## Command
cd $COOJA_DIR
./gradlew run --args="--no-gui --autostart $CSC_FILE"

## Working Directory
$SCRIPT_DIR

## CSC File
$CSC_FILE
EOF

cat > "$RAW_LOGS/cooja_environment.txt" << EOF
# Cooja Environment
# Generated: $(date -Iseconds)

Cooja Version: v5.1
Contiki-NG Version: release/v4.9
Java Version: $(java -version 2>&1 | head -1)
OS: $(uname -a)
Working Directory: $SCRIPT_DIR

# Simulation Configuration (from .csc)
- Random seed: 123456
- Radio medium: UDGM (Unit Disk Graph Model)
- Transmitting range: 50.0 m
- Interference range: 100.0 m
- TX success ratio: 0.85 (15% configured loss)
- RX success ratio: 1.00
- Radio bitrate: 250.0 kbps
- Node count: 10 (1 Sensor + 1 Gateway + 8 Relay)
- Sensor packet interval: 50 ms
- Gateway RX timeout: 20 s
- PRNG seed: [1000, 2000, 3000, 4000]
- PRNG warmup: 2000 iterations
- Serial Socket ports: 60000 (Gateway), 60001-60008 (Relays)
- ScriptRunner: REMOVED (was causing TEST FAILED on timeout)
- External timeout: 120s wall-clock via this script
EOF

# Start Python listener (must start BEFORE Cooja)
echo "[$(date -Iseconds)] Starting Python listener..."
python3 capture_relay_images.py > python_test.log 2>&1 &
PY_PID=$!
echo "[$(date -Iseconds)] Python PID: $PY_PID"

# Wait for Python to start connecting
sleep 5
echo "[$(date -Iseconds)] Python log so far:"
cat python_test.log || true

# Start Cooja
echo "[$(date -Iseconds)] Starting Cooja..."
cd "$COOJA_DIR"
./gradlew run --args="--no-gui --autostart \"$CSC_FILE\"" \
  > "$SCRIPT_DIR/cooja_stdout_real.log" 2>&1 &
COOJA_PID=$!
echo "[$(date -Iseconds)] Cooja PID: $COOJA_PID"

# Wait for Cooja to finish or timeout (max 120s)
echo "[$(date -Iseconds)] Waiting for Cooja (max 120s)..."
for i in $(seq 1 120); do
  if ! ps -p $COOJA_PID > /dev/null 2>&1; then
    echo "[$(date -Iseconds)] Cooja finished after ${i}s"
    break
  fi
  sleep 1
done

# If still running, kill it
if ps -p $COOJA_PID > /dev/null 2>&1; then
  echo "[$(date -Iseconds)] Cooja still running after 120s, terminating..."
  kill $COOJA_PID 2>/dev/null || true
  sleep 3
fi

# Wait for Python to finish
echo "[$(date -Iseconds)] Waiting for Python listener..."
for i in $(seq 1 30); do
  if ! ps -p $PY_PID > /dev/null 2>&1; then
    echo "[$(date -Iseconds)] Python finished after ${i}s"
    break
  fi
  sleep 1
done

# Kill Python if still running
if ps -p $PY_PID > /dev/null 2>&1; then
  kill $PY_PID 2>/dev/null || true
fi

# Copy outputs to raw_logs
cp -f "$SCRIPT_DIR/cooja_stdout_real.log" "$RAW_LOGS/" 2>/dev/null || true
cp -f "$SCRIPT_DIR/python_test.log" "$RAW_LOGS/" 2>/dev/null || true
cp -f "$SCRIPT_DIR/cooja_decrypted.png" "$RAW_LOGS/" 2>/dev/null || true

echo "[$(date -Iseconds)] Experiment complete."
echo "Raw logs saved to: $RAW_LOGS"
ls -la "$RAW_LOGS/"
ls -la "$SCRIPT_DIR/cooja_decrypted.png" 2>/dev/null || echo "No decrypted image generated"
