"""
Exact fields:
    s(r) = r^4 (1 - r)^4,   ψ(x,y) = s(x) s(y),
    u_ex = s_t(t) * (∂_y ψ, -∂_x ψ),   ω_ex = rot(u_ex),
    α_ex = (∂_y ω_ex, -∂_x ω_ex),      p_ex = 0.

Time factor:
    s_t(t) = e^{-t} (transient) or s_t(t) ≡ 1 (steady).

Forcing (with p_ex = 0 ⇒ ∇p = 0):
    f = ∂_t u_ex + ω_ex * (-u_ex,y, u_ex,x) + (1/Re) α_ex.

This makes (u_ex, ω_ex, α_ex, p_ex) solve the mixed system used by your scheme.
The helper returns the L2 velocity error at the final time.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Literal

from firedrake import *

from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper


@dataclass
class MMSResult:
    error: float
    dt: float
    N: int
    final_time: float
    time_factor: Literal["steady", "transient"]


TimeFactor = Literal["steady", "transient"]


def _space_shape(mesh):
    return SpatialCoordinate(mesh)


def _s(r):
    return r**4 * (1 - r)**4


def _grad_perp_scalar(phi):
    # (∂_y phi, -∂_x phi) — written componentwise
    return as_vector([phi.dx(1), -phi.dx(0)])


def _time_factor(expr_time, mode: TimeFactor):
    if mode == "steady":
        return Constant(1.0)
    if mode == "transient":
        return exp(-expr_time)
    raise ValueError(f"Unsupported time factor mode: {mode}")


def run_case(
    *,
    N: int,
    dt_value: float,
    final_time: float,
    Re_value: float,
    time_mode: TimeFactor,
    velocity_degree: int = 2,
    pressure_degree: int = 1,
    alpha_degree: int = 3,
    quadrilateral: bool = False,
    stages: int = 1,  # Gauss–Legendre stages (1=midpoint)
) -> MMSResult:
    """Run the scheme against the MMS and return the L2 velocity error."""

    # --- Mesh, spaces, time constants ----------------------------------------
    mesh = UnitSquareMesh(N, N, quadrilateral=quadrilateral)
    MC = MeshConstant(mesh)
    dt = MC.Constant(dt_value)
    time = MC.Constant(0.0)
    Re = Constant(Re_value)

    V = FunctionSpace(mesh, "CG", velocity_degree, variant="alfeld")
    Q = FunctionSpace(mesh, "DG", pressure_degree, variant="alfeld")
    W = FunctionSpace(mesh, "CG", alpha_degree, variant="alfeld")
    UP = MixedFunctionSpace([V, V, Q, V, V, Q, W])

    # Test functions
    v_x, v_y, q, gamma_x, gamma_y, delta, chi = TestFunctions(UP)
    v = as_vector([v_x, v_y])
    gamma = as_vector([gamma_x, gamma_y])

    # Unknown (mixed) state
    up = Function(UP, name="state")

    # UFL components for the variational form (tie symbolic dependence to `up`)
    u_x, u_y, p, alpha_x, alpha_y, beta, omega = split(up)
    u = as_vector([u_x, u_y])
    alpha = as_vector([alpha_x, alpha_y])

    # Concrete subfunctions for ICs / output / error
    u_x0, u_y0, p0, alpha_x0, alpha_y0, beta0, omega0 = up.subfunctions

    # --- Manufactured fields --------------------------------------------------
    x, y = _space_shape(mesh)
    psi = _s(x) * _s(y)                   # ψ = 0, Δψ = 0, ∇Δψ = 0 on ∂Ω
    spatial = _grad_perp_scalar(psi)      # (ψ_y, -ψ_x)

    s_t = _time_factor(time, time_mode)
    u_exact = s_t * spatial
    omega_exact = rot(u_exact)
    alpha_exact = _grad_perp_scalar(omega_exact)
    # p_exact ≡ 0

    # --- Forcing f = u_t + ω * (-u_y, u_x) + (1/Re) α ------------------------
    if time_mode == "transient":
        u_t_exact = -u_exact  # since s_t = e^{-t}
    else:
        u_t_exact = as_vector([0.0, 0.0])
    u_turn_exact = as_vector([-u_exact[1], u_exact[0]])
    forcing = u_t_exact + omega_exact * u_turn_exact + (1.0 / Re) * alpha_exact

    # --- Variational form -----------------------------------------------------
    def curl_scalar(phi):
        return as_vector([phi.dx(1), -phi.dx(0)])

    F = (
        (
            inner(Dt(u), v) * dx
            + inner(omega * as_vector([-u[1], u[0]]), v) * dx
            + (1 / Re) * inner(alpha, v) * dx
            - p * div(v) * dx
            - inner(forcing, v) * dx
        )
        - div(u) * q * dx
        + (
            inner(alpha, gamma) * dx
            - inner(rot(u), rot(gamma)) * dx
            - beta * div(gamma) * dx
        )
        - div(alpha) * delta * dx
        + (
            inner(curl_scalar(omega), curl_scalar(chi)) * dx
            - inner(alpha, curl_scalar(chi)) * dx
        )
    )

    # --- Boundary conditions  ----------
    bcs = [
        DirichletBC(UP.sub(0), 0, 1),
        DirichletBC(UP.sub(0), 0, 2),
        DirichletBC(UP.sub(1), 0, 3),
        DirichletBC(UP.sub(1), 0, 4),
        DirichletBC(UP.sub(3), 0, 1),
        DirichletBC(UP.sub(3), 0, 2),
        DirichletBC(UP.sub(4), 0, 3),
        DirichletBC(UP.sub(4), 0, 4),
        DirichletBC(UP.sub(6), 0, "on_boundary"),
    ]

    # --- Exact initial condition ---------------------------------------------
    u_x0.interpolate(u_exact[0]); u_y0.interpolate(u_exact[1])
    p0.assign(0.0)
    alpha_x0.interpolate(alpha_exact[0]); alpha_y0.interpolate(alpha_exact[1])
    beta0.assign(0.0)
    omega0.interpolate(omega_exact)

    # --- Time integration -----------------------------------------------------
    butcher_tableau = GaussLegendre(stages)
    solver_parameters = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}
    stepper = TimeStepper(F, butcher_tableau, time, dt, up, bcs=bcs, solver_parameters=solver_parameters)

    while float(time) < final_time - 1.0e-14:
        remaining = final_time - float(time)
        if remaining < float(dt):
            dt.assign(remaining)
        stepper.advance()
        time.assign(float(time) + float(dt))

    # --- Error at final time --------------------------------------------------
    uh = as_vector([u_x0, u_y0])
    err_sq = assemble(inner(uh - u_exact, uh - u_exact) * dx)
    error = math.sqrt(err_sq)

    return MMSResult(
        error=error,
        dt=float(dt_value),
        N=N,
        final_time=final_time,
        time_factor=time_mode,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Run a manufactured solution case.")
    parser.add_argument("--N", type=int, default=32, help="Elements per direction")
    parser.add_argument("--dt", type=float, default=2 ** -8, help="Initial time-step")
    parser.add_argument("--final-time", type=float, default=1.0, help="Final time T")
    parser.add_argument("--Re", type=float, default=4.0, help="Reynolds number")
    parser.add_argument("--time-mode", choices=["steady", "transient"], default="transient")
    parser.add_argument("--stages", type=int, default=1, help="Gauss–Legendre stages (1=midpoint)")
    return parser.parse_args()


def main():
    args = _parse_args()
    result = run_case(
        N=args.N,
        dt_value=args.dt,
        final_time=args.final_time,
        Re_value=args.Re,
        time_mode=args.time_mode,
        stages=args.stages,
    )
    print(f"L2 error = {result.error:.8e}")


if __name__ == "__main__":
    main()
