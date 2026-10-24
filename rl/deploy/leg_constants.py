# Leg kinematic constants extracted from the model (openroboticdog.xml), verified by sim/tests/test_leg_ik.py. Units: m, rad
LEGS = {
    "FR": dict(hip=(0.11653, -0.02450, -0.00023), x_off=0.01735, z_off=-0.00002, d=-0.09489, L1=0.10220, L2=0.10789, phi1_0=0.15020, phi2_0=0.79687),
    "FL": dict(hip=(0.11670, 0.02471, -0.00023), x_off=0.01713, z_off=-0.00022, d=0.09424, L1=0.10219, L2=0.10814, phi1_0=0.15053, phi2_0=0.79215),
    "RR": dict(hip=(-0.11577, -0.02450, -0.00023), x_off=-0.01735, z_off=-0.00002, d=-0.09489, L1=0.10220, L2=0.10762, phi1_0=0.15020, phi2_0=0.79625),
    "RL": dict(hip=(-0.11560, 0.02471, -0.00023), x_off=-0.01757, z_off=-0.00022, d=0.09424, L1=0.10219, L2=0.10814, phi1_0=0.15053, phi2_0=0.79215),
}
