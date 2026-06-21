# Probabilistic HEA model — P(HEA) per composition
# Three-layer architecture:
#   1. GP posterior (src/gp/) — calibrated binary mixing enthalpy
#   2. features.py            — 8 multicomponent features (mu_dH, sigma_dH, pen_*, delta, dS_R, T_m)
#   3. calibration.py         — logistic regression on experimental HEA outcomes
#   4. predict.py             — single entry point: composition -> P(HEA)
