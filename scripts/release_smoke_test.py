"""Release smoke test -- validates a freshly installed copulax wheel.

Run from a fresh venv after `pip install dist/copulax-*.whl`:

    python scripts/release_smoke_test.py

Exercises one representative API call from each public subpackage so we
catch packaging regressions (missing files, broken imports, namespace
collisions) before promoting a wheel to PyPI. This is intentionally
*much* lighter than the in-tree pytest suite -- it's an artefact-level
sanity check, not a correctness check. Mathematical correctness is
verified by the test suite that runs against the editable install.
"""

import sys


def section(name: str) -> None:
    print(f"\n[{name}]")


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import scipy.stats as sps

    # ------ 1. import surface ----------------------------------------
    section("imports")
    import copulax  # noqa: F401
    from copulax import univariate, multivariate, copulas, preprocessing
    from copulax import special as cs  # noqa: F401
    from copulax import stats as cstats  # noqa: F401
    print(f"  copulax {copulax.__version__} imported, all 5 subpackages resolved")

    # ------ 2. univariate --------------------------------------------
    section("univariate")
    from copulax.univariate import normal, student_t
    data = 1.5 + 2.0 * jax.random.normal(jax.random.PRNGKey(0), (1000,))
    fit = normal.fit(data)
    mu_hat = float(fit.params["mu"])
    sig_hat = float(fit.params["sigma"])
    assert abs(mu_hat - 1.5) < 0.2, f"normal mu recovery: {mu_hat}"
    assert abs(sig_hat - 2.0) < 0.2, f"normal sigma recovery: {sig_hat}"
    q = jnp.array([0.1, 0.5, 0.9])
    rt_err = float(jnp.max(jnp.abs(fit.cdf(fit.ppf(q)) - q)))
    assert rt_err < 1e-6, f"ppf round-trip err: {rt_err}"
    nu_data = jnp.asarray(sps.t.rvs(df=5, size=2000, random_state=1))
    nu_hat = float(student_t.fit(nu_data).params["nu"])
    assert 2.5 < nu_hat < 12.0, f"student_t nu recovery: {nu_hat}"
    print(f"  normal: mu={mu_hat:.3f} sigma={sig_hat:.3f}, ppf-rt err={rt_err:.1e}")
    print(f"  student_t: nu={nu_hat:.2f} (true 5.0)")

    # ------ 3. multivariate ------------------------------------------
    section("multivariate")
    from copulax.multivariate import mvt_normal, mvt_student_t  # noqa: F401
    d = 3
    mu = jnp.zeros((d, 1))
    sigma = jnp.eye(d)
    samples = mvt_normal.sample(size=500, params={"mu": mu, "sigma": sigma},
                                 key=jax.random.PRNGKey(2))
    assert samples.shape == (500, d) and jnp.all(jnp.isfinite(samples))
    logpdf = mvt_normal.logpdf(samples[:10], params={"mu": mu, "sigma": sigma})
    assert jnp.all(jnp.isfinite(logpdf))
    print(f"  mvt_normal sample {tuple(samples.shape)}, logpdf finite")

    # ------ 4. copulas (elliptical + Archimedean) --------------------
    section("copulas")
    from copulax.copulas import (
        gaussian_copula, clayton_copula, independence_copula,
    )
    from copulax.univariate import uniform as univ_uniform
    arch_params = {
        "marginals": tuple(
            (univ_uniform, {"a": jnp.array(0.0), "b": jnp.array(1.0)})
            for _ in range(2)
        ),
        "copula": {"theta": jnp.array(2.5)},
    }
    u = clayton_copula.copula_rvs(size=400, params=arch_params,
                                   key=jax.random.PRNGKey(4))
    assert u.shape == (400, 2) and jnp.all((u >= 0) & (u <= 1))
    pdf = clayton_copula.copula_pdf(u[:20], params=arch_params)
    assert jnp.all(jnp.isfinite(pdf)) and jnp.all(pdf >= 0)
    ind_lp = independence_copula.copula_logpdf(jnp.array([[0.3, 0.6, 0.5]]))
    assert jnp.allclose(ind_lp, 0.0, atol=1e-10)
    gauss_params = {
        "marginals": tuple(
            (univ_uniform, {"a": jnp.array(0.0), "b": jnp.array(1.0)})
            for _ in range(d)
        ),
        "copula": {"mu": jnp.zeros((d, 1)), "sigma": jnp.eye(d)},
    }
    u_g = gaussian_copula.copula_rvs(size=200, params=gauss_params,
                                      key=jax.random.PRNGKey(5))
    assert u_g.shape == (200, d) and jnp.all((u_g >= 0) & (u_g <= 1))
    print("  clayton_copula rvs+pdf, gaussian_copula rvs, independence_copula identity OK")

    # ------ 5. preprocessing -----------------------------------------
    section("preprocessing")
    from copulax.preprocessing import DataScaler
    raw = jnp.asarray(sps.norm.rvs(loc=10.0, scale=4.0, size=(200, 3),
                                    random_state=99))
    fitted, scaled = DataScaler("zscore").fit_transform(raw)
    assert float(jnp.max(jnp.abs(jnp.mean(scaled, axis=0)))) < 1e-10
    assert jnp.allclose(jnp.std(scaled, axis=0), 1.0, atol=1e-6)
    rt_err = float(jnp.max(jnp.abs(fitted.inverse_transform(scaled) - raw)))
    assert rt_err < 1e-10, f"DataScaler round-trip err: {rt_err}"
    print(f"  DataScaler(zscore) round-trip err {rt_err:.1e}")

    # ------ 6. special + stats ---------------------------------------
    section("special + stats")
    import scipy.special as ssp
    kv = float(cs.kv(jnp.array(1.5), jnp.array(3.0)))
    assert abs(kv - float(ssp.kv(1.5, 3.0))) / kv < 1e-6
    log_kv_big = float(cs.log_kv(jnp.array(2.0), jnp.array(200.0)))
    assert jnp.isfinite(log_kv_big)
    sk = float(cstats.skew(jnp.asarray(sps.gamma.rvs(a=2.0, size=500,
                                                     random_state=7))))
    assert jnp.isfinite(sk)
    print(f"  special.kv ok, log_kv(2, 200)={log_kv_big:.2f}, stats.skew={sk:.3f}")

    # ------ 7. save/load round-trip ----------------------------------
    section("save/load")
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.cpx")
        fit.save(path)
        loaded = copulax.load(path)
        a = fit.logpdf(data)
        b = loaded.logpdf(data)
        assert jnp.allclose(a, b)
    print("  fitted distribution save/load round-trip OK")

    print("\nALL SUBPACKAGE SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
