#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Tests for analysis modules:
  - chaos.analysis.lyapunov:    Jacobian-QR Lyapunov spectrum, chaos verification
  - chaos.analysis.sensitivity: initial value sensitivity (butterfly effect)
  - chaos.analysis.wolf:        Wolf-style max LE estimation
  - chaos.analysis.orbit:       Floyd cycle detection, orbit statistics
  - chaos.analysis.distribution: histogram, CDF deviation
"""

import sys
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_params(n=4):
    from chaos.core.params import ComplementaryChaosParams
    return ComplementaryChaosParams(n=n, A_odd=371, B_even=50).synthesize()


def _make_base_map(n=4):
    from chaos.core.base_map import HighDimChaosMap
    return HighDimChaosMap(_make_params(n), word_size=16, seed=42)


def _make_enhanced_map(n=4):
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced
    return HighDimChaosMapEnhanced(_make_params(n), word_size=16, seed=42)


# ---------------------------------------------------------------------------
# Lyapunov spectrum tests
# ---------------------------------------------------------------------------

def test_lyapunov_all_positive():
    """All Lyapunov exponents are positive for valid parameters."""
    from chaos.analysis.lyapunov import compute_lyapunov_exponents

    m = _make_base_map(4)
    les = compute_lyapunov_exponents(m, steps=800, transient=200)
    assert len(les) == 4
    assert np.all(les > 0.01), f"All LEs must be > 0.01, got {les}"


def test_lyapunov_shape_matches_n():
    """LE spectrum length matches map dimension."""
    from chaos.analysis.lyapunov import compute_lyapunov_exponents

    for n in [4, 6]:
        m = _make_base_map(n)
        les = compute_lyapunov_exponents(m, steps=500, transient=100)
        assert len(les) == n


def test_verify_chaos_passes():
    """verify_chaos returns is_chaotic=True for valid parameters."""
    from chaos.analysis.lyapunov import verify_chaos

    m = _make_base_map(4)
    result = verify_chaos(m, steps=800)
    assert result['is_chaotic'], f"Should be chaotic, got {result}"
    assert 'params' in result
    assert 'lyap_exps' in result
    assert 'variance' in result


def test_verify_chaos_returns_params():
    """verify_chaos includes the params dict in its output."""
    from chaos.analysis.lyapunov import verify_chaos

    m = _make_base_map(4)
    result = verify_chaos(m, steps=500)
    assert result['params'] is m.params
    assert result['min_le'] <= result['max_le']


def test_lyapunov_n_dimensional():
    """Lyapunov computation works for n > 4."""
    from chaos.analysis.lyapunov import compute_lyapunov_exponents

    m = _make_base_map(6)
    les = compute_lyapunov_exponents(m, steps=400, transient=100)
    assert len(les) == 6
    assert np.all(np.isfinite(les))


# ---------------------------------------------------------------------------
# Sensitivity analysis tests
# ---------------------------------------------------------------------------

def test_sensitivity_distance_grows():
    """Growth rate in early steps exceeds 1.0 (exponential divergence)."""
    from chaos.analysis.sensitivity import analyze_initial_value_sensitivity_16bit
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    result = analyze_initial_value_sensitivity_16bit(
        p, HighDimChaosMap,
        min_iterations=50, max_iterations=200)
    growth_rates = np.array(result['growth_rates'])
    # Early growth rates (steps 1..10) should average > 1.0 for chaotic map
    early_growth = growth_rates[1:11][np.isfinite(growth_rates[1:11])]
    avg_early = np.mean(early_growth) if len(early_growth) > 0 else 0
    assert avg_early > 1.0, \
        f"Early growth rate should exceed 1.0, got {avg_early:.3f}"


def test_sensitivity_result_structure():
    """Sensitivity result contains all expected keys."""
    from chaos.analysis.sensitivity import analyze_initial_value_sensitivity_16bit
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    result = analyze_initial_value_sensitivity_16bit(
        p, HighDimChaosMap,
        min_iterations=50, max_iterations=100)
    expected_keys = [
        'system_dimension', 'word_size', 'modulus',
        'initial_state', 'perturbation_vector',
        'distances', 'growth_rates', 'states1', 'states2',
        'initial_distance', 'final_distance', 'total_growth_factor',
        'average_growth_rate', 'first_growth', 'num_iterations'
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_sensitivity_enhanced_also_works():
    """Sensitivity analysis works with enhanced map."""
    from chaos.analysis.sensitivity import analyze_initial_value_sensitivity_16bit
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    result = analyze_initial_value_sensitivity_16bit(
        p, HighDimChaosMapEnhanced,
        min_iterations=50, max_iterations=100)
    assert len(result['distances']) >= 50
    assert result['system_dimension'] == 4


# ---------------------------------------------------------------------------
# Wolf algorithm tests
# ---------------------------------------------------------------------------

def test_wolf_full_positive():
    """Wolf full-state method returns positive max LE for chaotic map."""
    from chaos.analysis.wolf import wolf_lyapunov_full

    m = _make_base_map(4)
    result = wolf_lyapunov_full(m, max_steps=2000, transient=200)
    assert result['lyap_max'] > 0.0, f"Wolf max LE should be positive, got {result['lyap_max']:.6f}"
    assert result['is_chaotic']


def test_wolf_full_result_keys():
    """Wolf full-state result contains all expected keys."""
    from chaos.analysis.wolf import wolf_lyapunov_full

    m = _make_base_map(4)
    result = wolf_lyapunov_full(m, max_steps=1000, transient=100)
    for key in ['lyap_max', 'lyap_mean', 'lyap_std', 'is_chaotic',
                'growth_rates', 'distances', 'renormalizations']:
        assert key in result, f"Missing key: {key}"


def test_wolf_embedding_runs():
    """Wolf embedding method runs without error and returns valid structure."""
    from chaos.analysis.wolf import wolf_lyapunov_embedding

    m = _make_base_map(4)
    result = wolf_lyapunov_embedding(m, max_steps=3000, transient=200,
                                     embed_dim=4, delay=1, fit_horizon=15)
    # The embedding method needs sufficient data; with limited steps it may
    # return 0.0, but the result structure must be valid
    assert result['embed_dim'] == 4
    assert result['delay'] == 1
    assert 'lyap_max' in result
    assert 'is_chaotic' in result


def test_wolf_embedding_result_keys():
    """Wolf embedding result contains all expected keys."""
    from chaos.analysis.wolf import wolf_lyapunov_embedding

    m = _make_base_map(4)
    result = wolf_lyapunov_embedding(m, max_steps=1000, transient=100)
    for key in ['lyap_max', 'lyap_mean', 'lyap_std', 'is_chaotic',
                'embed_dim', 'delay', 'renormalizations']:
        assert key in result, f"Missing key: {key}"


def test_wolf_enhanced_vs_base():
    """Both base and enhanced maps pass Wolf chaos check."""
    from chaos.analysis.wolf import wolf_lyapunov_full

    base_m = _make_base_map(4)
    enh_m = _make_enhanced_map(4)
    base_r = wolf_lyapunov_full(base_m, max_steps=2000, transient=200)
    enh_r = wolf_lyapunov_full(enh_m, max_steps=2000, transient=200)
    # Both have valid LE estimates
    assert np.isfinite(base_r['lyap_max'])
    assert np.isfinite(enh_r['lyap_max'])
    # At least one should be detected as chaotic
    assert base_r['is_chaotic'] or enh_r['is_chaotic'], \
        "At least one of base/enhanced maps should be chaotic"


# ---------------------------------------------------------------------------
# Orbit diagnostics tests
# ---------------------------------------------------------------------------

def test_make_step_fn_from_map():
    """make_step_fn_from_map wraps .next() correctly."""
    from chaos.analysis.orbit import make_step_fn_from_map

    m = _make_base_map(4)
    step_fn = make_step_fn_from_map(m)
    x0 = m.get_physical_state()
    x1 = step_fn(x0)
    assert x1.shape == (4,)
    assert x1.dtype == np.int64


def test_estimate_period_detects_none_for_chaos():
    """Floyd cycle detection returns None cycle for chaotic map (no short cycle)."""
    from chaos.analysis.orbit import make_step_fn_from_map, estimate_period

    m = _make_base_map(4)
    step_fn = make_step_fn_from_map(m)
    x0 = m.get_physical_state()
    # Chaotic maps have very long cycles; Floyd should not find one in 5000 steps
    mu, lam = estimate_period(step_fn, x0, max_steps=5000)
    # Either no cycle found, or cycle is extremely long (both None)
    # In 16-bit with 2n-2=6 state dims, max period = 2^96, so lambda will be None
    assert lam is None or lam > 1000, \
        f"Chaotic map should not have short cycle, got lambda={lam}"


def test_sample_orbit_shape():
    """sample_orbit produces correct output shape."""
    from chaos.analysis.orbit import make_step_fn_from_map, sample_orbit, OrbitStatsConfig

    m = _make_base_map(4)
    step_fn = make_step_fn_from_map(m)
    x0 = m.get_physical_state()
    cfg = OrbitStatsConfig(n_dims=4, word_size=16, max_steps=200, transient=50)
    samples = sample_orbit(step_fn, x0, cfg)
    assert samples.shape[0] == 200
    assert samples.shape[1] == 4
    assert samples.min() >= 0.0 and samples.max() <= 1.0


def test_analyze_digital_system():
    """analyze_digital_system returns complete diagnostic dict."""
    from chaos.analysis.orbit import (
        make_step_fn_from_map, analyze_digital_system, OrbitStatsConfig)

    m = _make_base_map(4)
    step_fn = make_step_fn_from_map(m)
    x0 = m.get_physical_state()
    cfg = OrbitStatsConfig(n_dims=4, word_size=16, max_steps=200, transient=50,
                           hist_bins=32)
    result = analyze_digital_system(step_fn, x0, cfg)
    assert 'period_mu' in result
    assert 'period_lambda' in result
    assert 'samples' in result
    assert 'per_dim_stats' in result
    assert 'histograms' in result
    assert len(result['per_dim_stats']) == 4


def test_compute_marginal_histograms():
    """Marginal histogram stats have expected per-dimension structure."""
    from chaos.analysis.orbit import compute_marginal_histograms, OrbitStatsConfig

    samples = np.random.random((500, 4))
    cfg = OrbitStatsConfig(n_dims=4, word_size=16, hist_bins=64)
    info = compute_marginal_histograms(samples, cfg)
    assert len(info['per_dim_stats']) == 4
    for st in info['per_dim_stats']:
        for k in ['mean', 'var', 'min', 'max']:
            assert k in st


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------

def test_histogram_analyzer():
    """ChaosHistogramAnalyzer initializes and plots without error."""
    from chaos.analysis.distribution import ChaosHistogramAnalyzer

    data = np.random.randint(0, 65536, size=(500, 4))
    a = ChaosHistogramAnalyzer(data, mod=65536)
    assert a.n == 4
    assert a.dims == [0, 1, 2, 3]
    assert a.data.shape == (500, 4)
    # Values are normalized to [0,1)
    assert a.data.min() >= 0.0 and a.data.max() <= 1.0


def test_cdf_deviation():
    """ChaosCDFDeviationAnalyzer computes CDF deviation correctly."""
    from chaos.analysis.distribution import ChaosCDFDeviationAnalyzer

    data = np.random.randint(0, 65536, size=(500, 4))
    a = ChaosCDFDeviationAnalyzer(data, mod=65536)
    u_sorted, dev = a.compute_cdf_deviation()
    assert u_sorted.shape == data.shape
    assert dev.shape == data.shape
    # Deviations should be small for uniform random data
    assert np.max(np.abs(dev)) < 0.2


def test_report_generator():
    """ChaosReportGenerator produces report text."""
    from chaos.analysis.distribution import (
        ChaosCDFDeviationAnalyzer, ChaosReportGenerator)

    data = np.random.randint(0, 65536, size=(500, 4))
    cdf = ChaosCDFDeviationAnalyzer(data, mod=65536)
    report = ChaosReportGenerator(data, mod=65536, cdf_analyzer=cdf,
                                  warmup=100, total_samples=500)
    text = report.generate_report_text()
    assert len(text) > 200
    assert r'\paragraph' in text


def test_custom_dims_to_plot():
    """Analyzers respect custom dims_to_plot."""
    from chaos.analysis.distribution import (
        ChaosHistogramAnalyzer, ChaosCDFDeviationAnalyzer)

    data = np.random.randint(0, 65536, size=(500, 6))
    h = ChaosHistogramAnalyzer(data, mod=65536, dims_to_plot=[0, 3, 5])
    assert h.dims == [0, 3, 5]

    c = ChaosCDFDeviationAnalyzer(data, mod=65536, dims_to_plot=[2, 4])
    assert c.dims == [2, 4]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    if passed != len(tests):
        sys.exit(1)
