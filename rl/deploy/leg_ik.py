"""Analytic leg inverse kinematics. Foot target (body frame, relative to the hip body origin) → (hip, thigh, calf) angles.

Conventions (checked against the model, 2026-10-22): hip rotates about +x, thigh and calf about −y (positive = leg forward).
Link angle φ is "measured from straight down (−z) toward +x". Thigh link φ1 = phi1_0 + q_thigh,
calf link φ2 = phi2_0 + q_thigh + q_calf. Knee bend k = φ2 − φ1 = (phi2_0 − phi1_0) + q_calf ≥ 0 (knee backward).
"""
import math

from leg_constants import LEGS


def leg_fk(leg, q):
    """Forward kinematics for verification: (hip, thigh, calf) → foot position (relative to hip body origin, body frame)"""
    c = LEGS[leg]
    qh, qt, qc = q
    phi1 = c["phi1_0"] + qt
    phi2 = c["phi2_0"] + qt + qc
    x = c["x_off"] + c["L1"] * math.sin(phi1) + c["L2"] * math.sin(phi2)
    zl = c["z_off"] - (c["L1"] * math.cos(phi1) + c["L2"] * math.cos(phi2))   # vertical coordinate within the leg plane
    yl = c["d"]
    y = yl * math.cos(qh) - zl * math.sin(qh)
    z = yl * math.sin(qh) + zl * math.cos(qh)
    return x, y, z


def leg_ik(leg, target):
    """Foot target (x, y, z) → (hip, thigh, calf). If unreachable, returns the closest pose."""
    c = LEGS[leg]
    x, y, z = target
    r2 = y * y + z * z
    L = math.sqrt(max(r2 - c["d"] ** 2, 1e-8))
    q_hip = math.atan2(y, -z) - math.atan2(c["d"], L)
    xp = x - c["x_off"]
    Lp = L + c["z_off"]                 # in-plane vertical distance from the thigh axis to the foot
    D = math.hypot(xp, Lp)
    reach = c["L1"] + c["L2"]
    if D > reach - 1e-4:
        D = reach - 1e-4
    cos_k = (D * D - c["L1"] ** 2 - c["L2"] ** 2) / (2 * c["L1"] * c["L2"])
    k = math.acos(max(-1.0, min(1.0, cos_k)))
    phi1 = math.atan2(xp, Lp) - math.atan2(c["L2"] * math.sin(k), c["L1"] + c["L2"] * math.cos(k))
    q_thigh = phi1 - c["phi1_0"]
    q_calf = k - (c["phi2_0"] - c["phi1_0"])
    return q_hip, q_thigh, q_calf
