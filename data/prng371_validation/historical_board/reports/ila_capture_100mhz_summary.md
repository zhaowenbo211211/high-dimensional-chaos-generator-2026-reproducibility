# ILA Capture Summary — 100 MHz ACZ702 Clean Rerun

**Capture time**: 2026-Jun-23 16:18:11 CST  
**Source**: Real ACZ702 FPGA via JTAG (`__HW_SERVER_ENDPOINT__/xilinx_tcf/Digilent/__JTAG_SERIAL__`)  
**Capture file**: `prng_experiment/fpga_physical_validation/step1_clean_rerun_acz702_100mhz/capture/ila_capture_100mhz_clean_rerun.csv`  
**Capture SHA-256**: `a8a702832075e9a545771746c7fc13aa618e57d7d1d837b97d486f8c11663c42`  
**File size**: 875 KB  
**Total data samples**: 16384  
**ILA clock**: 100 MHz (same as internal PRNG clock)  

## ILA probe columns

| Column | Signal | Width | Notes |
|---|---|---|---|
| `dbg_state[4:0]` | FSM state | 5 bits | DATA ONLY |
| `dbg_heartbeat` | FSM-left-state-0 marker | 1 bit | **DATA AND TRIGGER** |
| `ila_probe2_diag[63:0]` | PRNG output / AXI diag bus | 64 bits | PRNG output after `cfg_done` |
| `dbg_axi_stat[3:0]` | AXI handshake status | 4 bits | DATA ONLY |
| `cfg_done` | Configuration-complete marker | 1 bit | DATA ONLY |
| `dbg_time[31:0]` | Free-running timestamp | 32 bits | DATA ONLY |

## Trigger configuration

- Trigger probe: `dbg_heartbeat`
- Trigger condition: `eq1'b1` (rising to 1)
- ILA armed at: 2026-Jun-23 16:17:56
- ILA triggered at: 2026-Jun-23 16:18:11

## Capture content

- Contains `cfg_done` rising edge at buffer sample 8215.
- Contains the one-cycle `0x0000000000000000` latency value immediately after `cfg_done`.
- Contains the 2000-output warmup interval after the latency cycle.
- Contains sufficient post-warmup outputs to extract at least 1000 64-bit words.

## Status

✅ Capture successful and ready for post-processing.
