"""
run_pipeline.py – Master orchestration script.

Runs the full evaluation pipeline for all trained models:
  1. Individual evaluation (evaluate_pro.py) for each available model.
  2. Comparative analysis (compare_models.py) across all methods.

Usage:
    python evaluation/run_pipeline.py [--models_root models/] [--n_episodes 10]
"""
