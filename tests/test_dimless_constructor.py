import numpy as np
import pytest

from fplanck.solver import FokkerPlanck


def harmonic_potential(x: np.ndarray) -> np.ndarray:
    return 0.5 * x**2


def test_physical_mode_requires_temperature_and_drag() -> None:
    with pytest.raises(ValueError, match="temperature is required"):
        FokkerPlanck(
            extent=10.0,
            resolution=0.1,
            drag=1.0,
        )

    with pytest.raises(ValueError, match="drag is required"):
        FokkerPlanck(
            extent=10.0,
            resolution=0.1,
            temperature=300.0,
        )


def test_physical_mode_rejects_diffusion_argument() -> None:
    with pytest.raises(ValueError, match="diffusion must not be provided"):
        FokkerPlanck(
            mode="physical",
            temperature=300.0,
            drag=1.0,
            diffusion=1.0,
            extent=10.0,
            resolution=0.1,
        )


def test_dimensionless_mode_defaults_to_unit_diffusion() -> None:
    fp = FokkerPlanck(
        mode="dimensionless",
        extent=10.0,
        resolution=0.1,
    )

    assert fp.mode == "dimensionless"
    assert fp.temperature is None
    assert np.allclose(fp.beta, 1.0)
    assert np.allclose(fp.diffusion[0], 1.0)


def test_dimensionless_mode_accepts_explicit_diffusion() -> None:
    fp = FokkerPlanck(
        mode="dimensionless",
        diffusion=0.5,
        extent=10.0,
        resolution=0.1,
    )

    assert np.allclose(fp.diffusion[0], 0.5)


def test_dimensionless_mode_rejects_temperature_and_drag() -> None:
    with pytest.raises(ValueError, match="temperature must not be provided"):
        FokkerPlanck(
            mode="dimensionless",
            temperature=300.0,
            extent=10.0,
            resolution=0.1,
        )

    with pytest.raises(ValueError, match="drag must not be provided"):
        FokkerPlanck(
            mode="dimensionless",
            drag=1.0,
            extent=10.0,
            resolution=0.1,
        )


def test_physical_mode_default_matches_explicit_physical_mode() -> None:
    fp_default = FokkerPlanck(
        temperature=300.0,
        drag=1.0,
        extent=10.0,
        resolution=0.1,
        potential=harmonic_potential,
    )

    fp_explicit = FokkerPlanck(
        mode="physical",
        temperature=300.0,
        drag=1.0,
        extent=10.0,
        resolution=0.1,
        potential=harmonic_potential,
    )

    assert fp_default.mode == "physical"
    assert fp_explicit.mode == "physical"
    assert np.allclose(fp_default.beta, fp_explicit.beta)
    assert np.allclose(fp_default.diffusion, fp_explicit.diffusion)
    assert np.allclose(fp_default.Rt, fp_explicit.Rt)
    assert np.allclose(fp_default.Lt, fp_explicit.Lt)
    assert np.allclose(fp_default.master_matrix.toarray(), fp_explicit.master_matrix.toarray())



def test_dimensionless_free_diffusion_rates_match_discrete_laplacian() -> None:
    diffusion = 0.5
    resolution = 0.1

    fp = FokkerPlanck(
        mode="dimensionless",
        diffusion=diffusion,
        extent=10.0,
        resolution=resolution,
    )

    expected_rate = diffusion / resolution**2

    assert np.allclose(fp.Rt[0, :-1], expected_rate)
    assert np.allclose(fp.Lt[0, 1:], expected_rate)
    assert fp.Lt[0, 0] == pytest.approx(0.0)
    assert fp.Rt[0, -1] == pytest.approx(0.0)



def test_dimensionless_quadratic_potential_produces_finite_rates() -> None:
    fp = FokkerPlanck(
        mode="dimensionless",
        extent=10.0,
        resolution=0.1,
        potential=harmonic_potential,
    )

    assert np.isfinite(fp.Rt).all()
    assert np.isfinite(fp.Lt).all()
    assert np.isfinite(fp.potential_values).all()