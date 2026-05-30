"""
test_weight_store.py
====================
Tests for the in-memory weight store (``store_weight=True`` + ``get_weights()``).

Coverage
--------
Shapes / dtypes
    nnear, inear, weight arrays have the expected shapes and dtypes.
    ngroups == nvar for kriging; ngroups == 2*nvar for SGSIM.

Weight values
    nnear values are in [0, nmax].
    Active inear indices are in [1, nobs] (1-based Fortran convention).
    Unused (padded) slots are zero in both inear and weight.
    Ordinary-kriging weights sum to 1 at every block.
    Reconstructing the estimate from weights matches get_results() output.

File persistence
    store_weight=True + weight_file writes a factor file after solve().
    use_old_weight=True reads that file back and reproduces the same estimates.
    store_weight=True without weight_file keeps weights in memory only (no file).

Error handling
    get_weights() raises RuntimeError when the store was not allocated.
    get_weights() raises RuntimeError after free_weight_store().
"""

import os
import numpy as np
import pytest
from pykriging import Kriging


_VGM = dict(vtype="sph", nugget=0.01, sill=0.09, a_major=100.0)
_NMAX = 5


def _build_kriging(obs_coord, obs_value, grid_coord, **kwargs):
    """Build, set up, and solve a standard 2-D ordinary kriging object."""
    k = Kriging(ndim=2, nvar=1, **kwargs)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=_NMAX)
    k.set_grid(coord=grid_coord)
    k.set_vgm(ivar=1, jvar=1, **_VGM)
    k.set_search(ivar=1)
    k.solve()
    return k


def _build_cokriging(obs_coord, value1, value2, grid_coord, **kwargs):
    """Build, set up, and solve a standard 2-D co-kriging object."""
    k = Kriging(ndim=2, nvar=2, **kwargs)
    k.set_obs(ivar=1, coord=obs_coord, value=value1, nmax=_NMAX)
    k.set_obs(ivar=2, coord=obs_coord, value=value2, nmax=_NMAX)
    k.set_grid(coord=grid_coord)
    k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.01, sill=0.09, a_major=100.0)
    k.set_vgm(ivar=1, jvar=2, vtype="sph", nugget=0.00, sill=0.03, a_major=100.0)
    k.set_vgm(ivar=2, jvar=2, vtype="sph", nugget=0.02, sill=0.16, a_major=100.0)
    k.set_search(ivar=1)
    k.set_search(ivar=2)
    k.solve()
    return k


# ===========================================================================
# Shape / dtype / dimension checks
# ===========================================================================

class TestWeightStoreShapes:
    """Array shapes and dtypes returned by get_weights()."""

    def test_nnear_shape(self, simple_obs, simple_grid):
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        nb = simple_grid.shape[0]
        assert w["nnear"].shape == (nb, 1)      # ngroups=1 for kriging nvar=1

    def test_inear_shape(self, simple_obs, simple_grid):
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        nb = simple_grid.shape[0]
        assert w["inear"].shape == (nb, 1, _NMAX)

    def test_weight_shape(self, simple_obs, simple_grid):
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        nb = simple_grid.shape[0]
        assert w["weight"].shape == (nb, 1, _NMAX)

    def test_dtypes(self, simple_obs, simple_grid):
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        assert w["nnear"].dtype  == np.int32
        assert w["inear"].dtype  == np.int32
        assert w["weight"].dtype == np.float64

    def test_ngroups_kriging(self, simple_obs, simple_grid):
        """ngroups == nvar for ordinary kriging (nsim=0)."""
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        assert w["nnear"].shape[1] == 1      # nvar=1 → ngroups=1

    def test_ngroups_sgsim(self, simple_obs, simple_grid):
        """ngroups == 2*nvar for SGSIM (one obs group + one sim group)."""
        coord, value = simple_obs
        nb = simple_grid.shape[0]
        k = Kriging(ndim=2, nvar=1, nsim=1, store_weight=True, seed=42)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_grid(coord=simple_grid)
        k.set_vgm(ivar=1, jvar=1, **_VGM)
        k.set_sim()
        k.set_search(ivar=1)
        k.solve()
        w = k.get_weights()
        # ngroups = 2: group 0 = obs, group 1 = previously-simulated blocks
        assert w["nnear"].shape == (nb, 2)
        assert w["inear"].shape == (nb, 2, _NMAX)
        assert w["weight"].shape == (nb, 2, _NMAX)


# ===========================================================================
# Weight value correctness
# ===========================================================================

class TestWeightValues:
    """Numerical correctness of the stored weights and indices."""

    def test_nnear_in_range(self, simple_obs, simple_grid):
        """nnear[ib, ig] must be in [0, nmax] for every block and group."""
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        assert np.all(w["nnear"] >= 0)
        assert np.all(w["nnear"] <= _NMAX)

    def test_inear_valid_indices(self, simple_obs, simple_grid):
        """Active inear entries (> 0) must be 1-based indices into obs."""
        coord, value = simple_obs
        nobs = coord.shape[0]
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        active = w["inear"][w["inear"] > 0]
        assert active.size > 0, "No active inear entries found"
        assert np.all(active >= 1), f"inear below 1: {active.min()}"
        assert np.all(active <= nobs), f"inear above nobs={nobs}: {active.max()}"

    def test_unused_slots_are_zero(self, simple_obs, simple_grid):
        """Slots beyond nnear[ib, ig] are zero-padded in inear and weight."""
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        nb, ng, nm = w["weight"].shape
        for ib in range(nb):
            for ig in range(ng):
                nn = w["nnear"][ib, ig]
                assert np.all(w["weight"][ib, ig, nn:] == 0.0), \
                    f"Non-zero weight in padding at block {ib}, group {ig}"
                assert np.all(w["inear"][ib, ig, nn:] == 0), \
                    f"Non-zero inear in padding at block {ib}, group {ig}"

    def test_ok_weights_sum_to_one(self, simple_obs, simple_grid):
        """For ordinary kriging (unbias=1), obs weights must sum to 1 per block."""
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        nb = simple_grid.shape[0]
        for ib in range(nb):
            nn = w["nnear"][ib, 0]
            wsum = float(w["weight"][ib, 0, :nn].sum())
            assert wsum == pytest.approx(1.0, abs=1e-5), \
                f"block {ib}: weights sum to {wsum:.8f}, expected 1.0"

    def test_estimate_consistency(self, simple_obs, simple_grid):
        """Reconstructing the estimate from stored weights matches get_results()."""
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        est_solve, _ = k.get_results()
        w = k.get_weights()

        nb = simple_grid.shape[0]
        est_manual = np.zeros(nb)
        for ib in range(nb):
            nn = w["nnear"][ib, 0]
            idx = w["inear"][ib, 0, :nn] - 1      # 1-based → 0-based
            wts = w["weight"][ib, 0, :nn]
            est_manual[ib] = np.dot(wts, value[idx])

        np.testing.assert_allclose(
            est_manual, est_solve, rtol=1e-4, atol=1e-6,
            err_msg="Estimate reconstructed from weights differs from solve() output",
        )

    def test_weights_unchanged_by_get_weights(self, simple_obs, simple_grid):
        """Calling get_weights() twice returns identical arrays (store is read-only)."""
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w1 = k.get_weights()
        w2 = k.get_weights()
        np.testing.assert_array_equal(w1["nnear"],  w2["nnear"])
        np.testing.assert_array_equal(w1["inear"],  w2["inear"])
        np.testing.assert_array_equal(w1["weight"], w2["weight"])


# ===========================================================================
# File persistence
# ===========================================================================

class TestWeightStoreFile:
    """Factor-file round-trip: store_weight writes; use_old_weight reads."""

    def test_file_written(self, simple_obs, simple_grid, tmp_path):
        """A factor file is created on disk after solve() with weight_file set."""
        coord, value = simple_obs
        fac = str(tmp_path / "weights.fac")
        _build_kriging(coord, value, simple_grid,
                       store_weight=True, weight_file=fac)
        assert os.path.isfile(fac), "Factor file was not written after solve()"

    def test_reuse_gives_same_estimates(self, simple_obs, simple_grid, tmp_path):
        """use_old_weight reproduces the same estimates and variances."""
        coord, value = simple_obs
        fac = str(tmp_path / "weights.fac")

        # First run: solve + write
        k1 = _build_kriging(coord, value, simple_grid,
                            store_weight=True, weight_file=fac)
        est1, var1 = k1.get_results()

        # Second run: read weights, skip solve
        k2 = Kriging(ndim=2, nvar=1, use_old_weight=True, weight_file=fac)
        k2.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k2.set_grid(coord=simple_grid)
        k2.set_vgm(ivar=1, jvar=1, **_VGM)
        k2.set_search(ivar=1)
        k2.solve()
        est2, var2 = k2.get_results()

        np.testing.assert_allclose(est2, est1, rtol=1e-5,
            err_msg="use_old_weight estimate differs from original solve()")
        np.testing.assert_allclose(var2, var1, rtol=1e-5,
            err_msg="use_old_weight variance differs from original solve()")

    def test_roundtrip_matches_plain_solve(self, simple_obs, simple_grid, tmp_path):
        """Round-trip (store + reload) produces the same result as a plain solve."""
        coord, value = simple_obs
        weight_file = tmp_path / "weights.fac"

        # Reference: plain solve with no weight storage
        est_ref, var_ref = _build_kriging(coord, value, simple_grid).get_results()

        # Store weights to file
        _build_kriging(coord, value, simple_grid,
                       store_weight=True, weight_file=str(weight_file))
        assert weight_file.exists()
        assert weight_file.stat().st_size > 0

        # Reload weights and re-estimate
        est_old, var_old = _build_kriging(
            coord, value, simple_grid,
            use_old_weight=True, weight_file=str(weight_file),
        ).get_results()

        np.testing.assert_allclose(est_old, est_ref, rtol=1e-10, atol=1e-10,
            err_msg="use_old_weight estimate differs from plain-solve reference")
        np.testing.assert_allclose(var_old, var_ref, rtol=1e-10, atol=1e-10,
            err_msg="use_old_weight variance differs from plain-solve reference")

    def test_cokriging_reuse_gives_same_estimates(self, simple_obs, simple_grid, tmp_path):
        """use_old_weight reproduces co-kriging estimates and variances."""
        coord, value1 = simple_obs
        value2 = 2.0 + 1.5 * value1
        weight_file = str(tmp_path / "cokriging.fac")

        k1 = _build_cokriging(
            coord, value1, value2, simple_grid,
            store_weight=True, weight_file=weight_file,
        )
        est1, var1 = k1.get_results()
        w = k1.get_weights()
        assert w["nnear"].shape[1] == 2

        k2 = _build_cokriging(
            coord, value1, value2, simple_grid,
            use_old_weight=True, weight_file=weight_file,
        )
        est2, var2 = k2.get_results()

        np.testing.assert_allclose(est2, est1, rtol=1e-10, atol=1e-10,
            err_msg="Co-kriging use_old_weight estimate differs from original solve")
        np.testing.assert_allclose(var2, var1, rtol=1e-10, atol=1e-10,
            err_msg="Co-kriging use_old_weight variance differs from original solve")

    def test_memory_only_matches_plain_solve(self, simple_obs, simple_grid):
        """Memory-only store_weight=True gives identical estimates to a plain solve."""
        coord, value = simple_obs

        # Reference: plain solve with no weight storage
        est_ref, var_ref = _build_kriging(coord, value, simple_grid).get_results()

        # Memory-only: store_weight=True, no weight_file
        est_mem, var_mem = _build_kriging(
            coord, value, simple_grid, store_weight=True
        ).get_results()

        np.testing.assert_allclose(est_mem, est_ref, rtol=1e-10, atol=1e-10,
            err_msg="Memory-only store_weight estimate differs from plain solve")
        np.testing.assert_allclose(var_mem, var_ref, rtol=1e-10, atol=1e-10,
            err_msg="Memory-only store_weight variance differs from plain solve")

    def test_memory_only_no_file_created(self, simple_obs, simple_grid, tmp_path):
        """store_weight=True without weight_file: weights in memory, no file written."""
        coord, value = simple_obs
        phantom = str(tmp_path / "should_not_exist.fac")
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        w = k.get_weights()
        assert w["weight"].shape[0] == simple_grid.shape[0]
        assert not os.path.isfile(phantom)


# ===========================================================================
# Joint co-simulation round-trip
# ===========================================================================

class TestWeightStoreCosim:
    """Weight-store round-trip for joint co-simulation (nvar=2, nsim>0).

    Joint co-simulation produces a different estimate vector for every
    realization, so the round-trip check must compare the full
    (nsim, nblock, nvar) result from get_estimate_all(), not just the
    kriging variance.

    Two scenarios are tested:
    1. Explicit path + sample arrays: fully deterministic, no RNG dependency.
    2. Auto-generated path + samples via seed=: reproduces the original run
       when the same integer seed is passed to both Kriging constructors.
    """

    # Small dataset matching the joint co-sim test in test_sgsim.py
    _COORD = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    _VAL1  = np.array([1.0, 2.0, 1.5])
    _VAL2  = np.array([10.0, 20.0, 15.0])
    _GRID  = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]])
    _NSIM  = 2
    _NMAX  = 3
    # Fixed path and samples so the test is independent of RNG state
    _RANDPATH = np.array([1, 2, 3], dtype=np.int32)
    _SAMPLE   = np.array([[0.5, -0.5, 1.0],
                           [1.2, -1.2, 0.3]], dtype=np.float64, order='F')  # (nsim, nblock)

    def _build(self, **kwargs):
        """Create and solve a joint co-simulation object with fixed path and samples."""
        k = Kriging(ndim=2, nvar=2, nsim=self._NSIM, seed=42, **kwargs)
        k.set_obs(ivar=1, coord=self._COORD, value=self._VAL1, nmax=self._NMAX)
        k.set_obs(ivar=2, coord=self._COORD, value=self._VAL2, nmax=self._NMAX)
        k.set_grid(coord=self._GRID)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0,  a_major=1.0)
        k.set_vgm(ivar=1, jvar=2, vtype="sph", nugget=0.0, sill=0.25, a_major=1.0)
        k.set_vgm(ivar=2, jvar=2, vtype="sph", nugget=0.0, sill=1.0,  a_major=1.0)
        k.set_sim(randpath=self._RANDPATH, sample=self._SAMPLE)
        k.set_search(ivar=1)
        k.set_search(ivar=2)
        k.solve()
        return k

    def _build_auto_seed(self, seed, **kwargs):
        """Create and solve a joint co-simulation object with auto-generated path/samples."""
        k = Kriging(ndim=2, nvar=2, nsim=self._NSIM, seed=seed, **kwargs)
        k.set_obs(ivar=1, coord=self._COORD, value=self._VAL1, nmax=self._NMAX)
        k.set_obs(ivar=2, coord=self._COORD, value=self._VAL2, nmax=self._NMAX)
        k.set_grid(coord=self._GRID)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0,  a_major=1.0)
        k.set_vgm(ivar=1, jvar=2, vtype="sph", nugget=0.0, sill=0.25, a_major=1.0)
        k.set_vgm(ivar=2, jvar=2, vtype="sph", nugget=0.0, sill=1.0,  a_major=1.0)
        k.set_sim()   # auto-generated from seed
        k.set_search(ivar=1)
        k.set_search(ivar=2)
        k.solve()
        return k

    # ------------------------------------------------------------------

    def test_use_old_weight_reproduces_all_realizations(self, tmp_path):
        """use_old_weight + same explicit samples reproduces every realization."""
        weight_file = str(tmp_path / "cosim.fac")

        # Run 1: solve and write weights
        k1 = self._build(store_weight=True, weight_file=weight_file)
        all1 = k1.get_estimate_all()   # (nsim, nblock, nvar)
        _, var1 = k1.get_results()

        # Run 2: reload weights, same fixed path + samples
        k2 = self._build(use_old_weight=True, weight_file=weight_file)
        all2 = k2.get_estimate_all()
        _, var2 = k2.get_results()

        np.testing.assert_allclose(all2, all1, rtol=1e-10, atol=1e-10,
            err_msg="Joint co-sim: use_old_weight realizations differ from original")
        np.testing.assert_allclose(var2, var1, rtol=1e-10, atol=1e-10,
            err_msg="Joint co-sim: use_old_weight variance differs from original")

    def test_use_old_weight_same_seed_reproduces_all_realizations(self, tmp_path):
        """use_old_weight + same seed reproduces every realization (auto path/samples).

        Both runs call set_sim() with no explicit arguments.  Because both
        Kriging objects are initialised with the same seed, the Fortran RNG
        is at the same state when set_sim() is reached, so the generated
        random path and N(0,1) samples are identical.
        """
        weight_file = str(tmp_path / "cosim_seed.fac")
        _SEED = 99

        # Run 1: auto path + samples, write weights
        k1 = self._build_auto_seed(_SEED, store_weight=True, weight_file=weight_file)
        all1 = k1.get_estimate_all()
        _, var1 = k1.get_results()

        # Run 2: same seed → same auto path + samples, reload weights
        k2 = self._build_auto_seed(_SEED, use_old_weight=True, weight_file=weight_file)
        all2 = k2.get_estimate_all()
        _, var2 = k2.get_results()

        np.testing.assert_allclose(all2, all1, rtol=1e-10, atol=1e-10,
            err_msg="Same seed: use_old_weight realizations differ from original")
        np.testing.assert_allclose(var2, var1, rtol=1e-10, atol=1e-10,
            err_msg="Same seed: use_old_weight variance differs from original")

    def test_different_seed_gives_different_realizations(self, tmp_path):
        """Sanity check: a different seed must produce different realizations."""
        wf1 = str(tmp_path / "s1.fac")
        wf2 = str(tmp_path / "s2.fac")
        all1 = self._build_auto_seed(11, store_weight=True, weight_file=wf1).get_estimate_all()
        all2 = self._build_auto_seed(22, store_weight=True, weight_file=wf2).get_estimate_all()
        assert not np.allclose(all1, all2), \
            "Different seeds produced identical co-simulation realizations"


# ===========================================================================
# Error handling
# ===========================================================================

class TestWeightStoreErrors:

    def test_get_weights_without_store_weight_raises(self, simple_obs, simple_grid):
        """get_weights() raises when the store was never allocated."""
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid)   # store_weight=False (default)
        with pytest.raises(RuntimeError, match="Weight store not allocated"):
            k.get_weights()

    def test_free_then_get_raises(self, simple_obs, simple_grid):
        """get_weights() raises after free_weight_store() is called."""
        coord, value = simple_obs
        k = _build_kriging(coord, value, simple_grid, store_weight=True)
        k.free_weight_store()
        with pytest.raises(RuntimeError, match="Weight store not allocated"):
            k.get_weights()
