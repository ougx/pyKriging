"""
test_spacetime_kriging.py
=========================
Tests for SpaceTimeKriging, spacetime_kriging(), and spacetime_cokriging().

These tests use synthetic data so they don't require the compiled Fortran
library at import time — the import of pykriging is guarded so that the
test collection step passes even without a compiled library.

Run:
    pytest tests/test_spacetime_kriging.py -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if the compiled library is not available
# ---------------------------------------------------------------------------
pytest.importorskip("pykriging", reason="compiled libkriging not found")
from pykriging import SpaceTimeKriging, spacetime_kriging, spacetime_cokriging


# ---------------------------------------------------------------------------
# Shared synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def obs_data_1var(rng):
    """200 observations in a 1000×1000×100 m domain over 50 years."""
    nobs = 200
    coord = rng.uniform([0, 0, 0], [1000, 1000, 100], size=(nobs, 3))
    time  = rng.uniform(0, 50, size=nobs)    # decimal years
    value = (np.sin(coord[:, 0] / 200) +
             np.cos(coord[:, 1] / 300) +
             0.01 * time +
             rng.normal(0, 0.1, nobs))
    return coord, value, time


@pytest.fixture(scope="module")
def grid_data(rng):
    """50 prediction points."""
    ngrid = 50
    coord = rng.uniform([0, 0, 0], [1000, 1000, 100], size=(ngrid, 3))
    time  = rng.uniform(0, 50, size=ngrid)
    return coord, time


@pytest.fixture(scope="module")
def obs_data_2var(rng):
    """Two correlated variables with 150 obs each."""
    nobs = 150
    c1 = rng.uniform([0, 0, 0], [1000, 1000, 100], size=(nobs, 3))
    t1 = rng.uniform(0, 50, size=nobs)
    v1 = np.sin(c1[:, 0] / 200) + 0.1 * rng.normal(size=nobs)

    c2 = rng.uniform([0, 0, 0], [1000, 1000, 100], size=(nobs, 3))
    t2 = rng.uniform(0, 50, size=nobs)
    v2 = 0.7 * np.sin(c2[:, 0] / 200) + 0.1 * rng.normal(size=nobs)

    return (c1, v1, t1), (c2, v2, t2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _basic_st_vgm(k, ivar=1, jvar=1):
    """Attach a simple sum-metric variogram to (ivar, jvar)."""
    k.set_vgm(ivar, jvar, vtype="sph", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100)
    k.set_vgm_temporal(ivar, jvar, vtype="exp", nugget=0, sill=0.5, at_k=10.0)
    k.set_vgm_joint_sills(ivar, jvar, 0.3)


# ===========================================================================
# 1. sum-metric, linear transform
# ===========================================================================
class TestSumMetricLinear:

    def test_basic_shape(self, obs_data_1var, grid_data):
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model(model="sum_metric", transform="linear", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20, maxdist=800, maxtlag=15)
        _basic_st_vgm(k)
        k.set_grid(gcoord, gtime)
        k.set_search(1)
        k.solve()
        est, var = k.get_results()

        assert est.shape == (len(gtime),), "estimate shape mismatch"
        assert var.shape == (len(gtime),), "variance shape mismatch"

    def test_variance_non_negative(self, obs_data_1var, grid_data):
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "linear", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20)
        _basic_st_vgm(k)
        k.set_grid(gcoord, gtime)
        k.set_search(1)
        k.solve()
        _, var = k.get_results()
        assert np.all(var >= 0), f"negative variance encountered: min={var.min():.6f}"

    def test_estimate_finite(self, obs_data_1var, grid_data):
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "linear", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20)
        _basic_st_vgm(k)
        k.set_grid(gcoord, gtime)
        k.set_search(1)
        k.solve()
        est, _ = k.get_results()
        assert np.all(np.isfinite(est)), "non-finite estimates found"

    def test_exact_match(self, obs_data_1var):
        """Predict at an observation location — estimate should equal observed value."""
        coord, value, time = obs_data_1var
        idx = 5

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "linear", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20)
        _basic_st_vgm(k)
        k.set_grid(coord[[idx]], time[[idx]])
        k.set_search(1)
        k.solve()
        est, var = k.get_results()

        assert abs(est[0] - value[idx]) < 1e-3, \
            f"exact match failed: est={est[0]:.4f} obs={value[idx]:.4f}"


# ===========================================================================
# 2. sum-metric, bounded transform
# ===========================================================================
class TestSumMetricBounded:

    def test_bounded_transform(self, obs_data_1var, grid_data):
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "bounded", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20)
        _basic_st_vgm(k)
        k.set_grid(gcoord, gtime)
        k.set_search(1)
        k.solve()
        est, var = k.get_results()

        assert np.all(var >= 0)
        assert np.all(np.isfinite(est))

    def test_maxtlag_reduces_neighbours(self, obs_data_1var, grid_data):
        """Tight maxtlag → more no-neighbour (NaN) blocks than wide maxtlag."""
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        # Wide lag: all obs are temporal neighbours → no NaN blocks
        k_wide = SpaceTimeKriging(nvar=1, neglect_error=True)
        k_wide.set_st_model("sum_metric", "bounded", at=10.0)
        k_wide.set_obs(1, coord, value, time, nmax=20, maxtlag=100.0)
        _basic_st_vgm(k_wide)
        k_wide.set_grid(gcoord, gtime)
        k_wide.set_search(1)
        k_wide.solve()
        est_wide, _ = k_wide.get_results()

        # Tight lag: spatial KD-tree neighbors often miss the temporal window
        k_tight = SpaceTimeKriging(nvar=1, neglect_error=True)
        k_tight.set_st_model("sum_metric", "bounded", at=10.0)
        k_tight.set_obs(1, coord, value, time, nmax=20, maxtlag=1.0)
        _basic_st_vgm(k_tight)
        k_tight.set_grid(gcoord, gtime)
        k_tight.set_search(1)
        k_tight.solve()
        est_tight, _ = k_tight.get_results()

        # Wide should have no NaN; tight may have some (gracefully handled)
        assert np.sum(np.isnan(est_wide)) == 0, "wide maxtlag should find neighbours for all blocks"
        assert np.sum(np.isnan(est_tight)) >= np.sum(np.isnan(est_wide)), \
            "tight maxtlag should produce at least as many NaN blocks as wide"


# ===========================================================================
# 3. product-sum model
# ===========================================================================
class TestProductSum:

    def test_product_sum(self, obs_data_1var, grid_data):
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("product_sum", "linear", at=10.0, k_ps=0.5)
        k.set_obs(1, coord, value, time, nmax=20)
        k.set_vgm(1, 1, vtype="sph", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100)
        k.set_vgm_temporal(1, 1, vtype="exp", nugget=0, sill=0.5, at_k=10.0)
        k.set_grid(gcoord, gtime)
        k.set_search(1)
        k.solve()
        est, var = k.get_results()

        assert est.shape == (len(gtime),)
        assert np.all(var >= 0)
        assert np.all(np.isfinite(est))


# ===========================================================================
# 4. Convenience function spacetime_kriging()
# ===========================================================================
class TestConvenienceFunction:

    def test_spacetime_kriging_sum_metric(self, obs_data_1var, grid_data):
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        est, var = spacetime_kriging(
            obs_coord=coord,
            obs_value=value,
            obs_time=time,
            grid_coord=gcoord,
            grid_time=gtime,
            spatial_spec=dict(vtype="sph", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100),
            temporal_spec=dict(vtype="exp", nugget=0, sill=0.5, at_k=10.0),
            joint_sills=[0.3],
            model="sum_metric",
            transform="linear",
            at=10.0,
            nmax=20,
        )
        assert est.shape == (len(gtime),)
        assert np.all(var >= 0)

    def test_spacetime_kriging_product_sum(self, obs_data_1var, grid_data):
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        est, var = spacetime_kriging(
            obs_coord=coord,
            obs_value=value,
            obs_time=time,
            grid_coord=gcoord,
            grid_time=gtime,
            spatial_spec=dict(vtype="exp", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100),
            temporal_spec=dict(vtype="exp", nugget=0, sill=0.5, at_k=10.0),
            joint_sills=[],       # not used for product_sum
            model="product_sum",
            transform="linear",
            at=10.0,
            alpha=1.0,
            k_ps=0.3,
            nmax=20,
        )
        assert est.shape == (len(gtime),)
        assert np.all(var >= 0)


# ===========================================================================
# 5. Cross-validation
# ===========================================================================
class TestCrossValidation:

    def test_cv_shape(self, obs_data_1var):
        coord, value, time = obs_data_1var

        k = SpaceTimeKriging(nvar=1, cross_validation=True)
        k.set_st_model("sum_metric", "linear", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20)
        _basic_st_vgm(k)
        k.set_grid_cv()
        k.set_search(1)
        k.solve()
        est, var = k.get_results()

        assert est.shape == (len(value),)
        assert var.shape == (len(value),)
        assert np.all(var >= 0)


# ===========================================================================
# 6. Co-kriging (2 variables)
# ===========================================================================
class TestCokriging:

    def test_cokriging_shape(self, obs_data_2var, grid_data):
        (c1, v1, t1), (c2, v2, t2) = obs_data_2var
        gcoord, gtime = grid_data

        k = SpaceTimeKriging(nvar=2)
        k.set_st_model("sum_metric", "linear", at=10.0)
        k.set_obs(1, c1, v1, t1, nmax=20)
        k.set_obs(2, c2, v2, t2, nmax=20)

        # Auto-variograms
        k.set_vgm(1, 1, vtype="sph", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100)
        k.set_vgm_temporal(1, 1, vtype="exp", nugget=0, sill=0.5, at_k=10.0)
        k.set_vgm_joint_sills(1, 1, 0.3)

        k.set_vgm(2, 2, vtype="sph", nugget=0, sill=0.6, a_major=500, a_minor1=300, a_minor2=100)
        k.set_vgm_temporal(2, 2, vtype="exp", nugget=0, sill=0.4, at_k=10.0)
        k.set_vgm_joint_sills(2, 2, 0.2)

        # Cross-variogram (LMC: sill_12^2 <= sill_11 * sill_22)
        k.set_vgm(1, 2, vtype="sph", nugget=0, sill=0.3, a_major=500, a_minor1=300, a_minor2=100)
        k.set_vgm_temporal(1, 2, vtype="exp", nugget=0, sill=0.2, at_k=10.0)
        k.set_vgm_joint_sills(1, 2, 0.1)

        k.set_grid(gcoord, gtime)
        k.set_search(1)
        k.set_search(2)
        k.solve()
        est, var = k.get_results()

        assert est.shape == (len(gtime),)
        assert np.all(var >= 0)
        assert np.all(np.isfinite(est))

    def test_cokriging_convenience(self, obs_data_2var, grid_data):
        (c1, v1, t1), (c2, v2, t2) = obs_data_2var
        gcoord, gtime = grid_data

        est, var = spacetime_cokriging(
            obs_coords=[c1, c2],
            obs_values=[v1, v2],
            obs_times=[t1, t2],
            grid_coord=gcoord,
            grid_time=gtime,
            spatial_specs={
                (1,1): dict(vtype="sph", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100),
                (2,2): dict(vtype="sph", nugget=0, sill=0.6, a_major=500, a_minor1=300, a_minor2=100),
                (1,2): dict(vtype="sph", nugget=0, sill=0.3, a_major=500, a_minor1=300, a_minor2=100),
            },
            temporal_specs={
                (1,1): dict(vtype="exp", nugget=0, sill=0.5, at_k=10.0),
                (2,2): dict(vtype="exp", nugget=0, sill=0.4, at_k=10.0),
                (1,2): dict(vtype="exp", nugget=0, sill=0.2, at_k=10.0),
            },
            joint_sills={
                (1,1): [0.3],
                (2,2): [0.2],
                (1,2): [0.1],
            },
            model="sum_metric",
            at=10.0,
            nmax=20,
        )
        assert est.shape == (len(gtime),)
        assert np.all(var >= 0)


# ===========================================================================
# 7. Nested variogram structures
# ===========================================================================
class TestNestedStructures:

    def test_nested_spatial(self, obs_data_1var, grid_data):
        """Nugget + spherical spatial, single exponential temporal."""
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "linear", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20)
        # Two spatial structures
        k.set_vgm(1, 1, vtype="nug", nugget=0.1, sill=0.0, a_major=1.0)
        k.set_vgm(1, 1, vtype="sph", nugget=0.0, sill=0.7, a_major=500, a_minor1=300, a_minor2=100)
        k.set_vgm_temporal(1, 1, vtype="exp", nugget=0, sill=0.5, at_k=10.0)
        k.set_vgm_joint_sills(1, 1, 0.0, 0.3)  # joint sill for nug=0, sph=0.3
        k.set_grid(gcoord, gtime)
        k.set_search(1)
        k.solve()
        est, var = k.get_results()
        assert np.all(var >= 0)
        assert np.all(np.isfinite(est))

    def test_nested_temporal(self, obs_data_1var, grid_data):
        """Single spherical spatial, nugget + exponential temporal."""
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "linear", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20)
        k.set_vgm(1, 1, vtype="sph", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100)
        k.set_vgm_temporal(1, 1, vtype="nug", nugget=0.05, sill=0.0, at_k=1.0)
        k.set_vgm_temporal(1, 1, vtype="exp", nugget=0.0, sill=0.45, at_k=10.0)
        k.set_vgm_joint_sills(1, 1, 0.3)
        k.set_grid(gcoord, gtime)
        k.set_search(1)
        k.solve()
        est, var = k.get_results()
        assert np.all(var >= 0)
        assert np.all(np.isfinite(est))


# ===========================================================================
# 8. SGSIM (primary variable)
# ===========================================================================
class TestSGSIM:

    def test_sgsim_shape(self, obs_data_1var, grid_data, rng):
        coord, value, time = obs_data_1var
        gcoord, gtime = grid_data
        nsim = 3

        k = SpaceTimeKriging(nvar=1, nsim=nsim, seed=123)
        k.set_st_model("sum_metric", "linear", at=10.0)
        k.set_obs(1, coord, value, time, nmax=20)
        _basic_st_vgm(k)
        k.set_grid(gcoord, gtime)
        k.set_sim()
        k.set_search(1)
        k.solve()
        sims, var = k.get_results()

        assert sims.shape == (nsim, len(gtime)), \
            f"SGSIM shape mismatch: {sims.shape}"
        assert np.all(var >= 0)
        # Simulations should differ across realisations
        assert not np.allclose(sims[0], sims[1]), "simulations should not be identical"
