"""
dapp_est: rule-based SM cap / gate estimator for a YOLO tenant co-located
with cuPHY under MPS. Pure Python (no numpy) in the decision path so the
same code runs in replay and live mode; numpy/matplotlib are only used by
the offline tools (replay plots, calibrate, evaluate).

Every constant that is not derived from the FAPI record itself lives in
rules.yaml / cells.yaml / profiles/*.yaml. Values that have not been
measured yet are tagged "provisional" there and listed in README.md.
"""
__version__ = "0.1"
