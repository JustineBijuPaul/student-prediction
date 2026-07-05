#!/usr/bin/env python3
"""
Student Performance Prediction Agent — main entry point.

Usage:
    python main.py --train              # Train models and save artifacts
    python main.py --predict            # Demo prediction on sample student
    python main.py --train --predict    # Full pipeline
    python main.py --dashboard          # Batch score local dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_train() -> None:
    """Train all models and persist best model + visualizations."""
    from src.train import train_all_models

    print("=" * 60)
    print("Student Performance Prediction Agent — Training Pipeline")
    print("=" * 60)
    result = train_all_models(save_artifacts=True)
    best = result["best"]
    print(f"\nTraining complete. Best model: {best['model']}")
    print(f"Test Accuracy: {best['accuracy']:.4f} | F1: {best['f1']:.4f}")


def cmd_predict() -> None:
    """Run demo inference through the AI Agent."""
    from src.predict import load_agent

    print("=" * 60)
    print("Student Performance Prediction Agent — Inference Demo")
    print("=" * 60)

    agent = load_agent()

    # Representative at-risk student profile
    sample_student = {
        "school": "GP",
        "sex": "M",
        "age": 17,
        "address": "U",
        "famsize": "LE3",
        "Pstatus": "T",
        "Medu": 1,
        "Fedu": 1,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "home",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": 1,
        "failures": 2,
        "schoolsup": "yes",
        "famsup": "no",
        "paid": "no",
        "activities": "no",
        "nursery": "yes",
        "higher": "yes",
        "internet": "no",
        "romantic": "no",
        "famrel": 3,
        "freetime": 3,
        "goout": 4,
        "Dalc": 2,
        "Walc": 2,
        "health": 3,
        "absences": 20,
        "G1": 8,
        "G2": 7,
        "course": "Math",
    }

    prediction = agent.predict(sample_student, student_id="demo_001")

    print("\n--- Agent Decision Output ---")
    print(f"Performance Category : {prediction.performance_category}")
    print(f"Confidence           : {prediction.confidence:.2%}")
    print(f"Risk Level           : {prediction.risk_level}")
    print(f"Intervention Priority: {prediction.intervention_priority}")
    print(f"\nClass Probabilities:")
    for label, prob in prediction.probabilities.items():
        print(f"  {label}: {prob:.2%}")
    print(f"\nRecommended Actions:")
    for i, rec in enumerate(prediction.recommendations, 1):
        print(f"  {i}. {rec}")


def cmd_dashboard() -> None:
    """Batch-score dataset and write teacher dashboard CSV."""
    from src.predict import load_agent
    from src.utils import get_paths, load_student_dataset

    agent = load_agent()
    df = load_student_dataset()
    dashboard = agent.batch_predict_df(df)

    out_path = get_paths()["outputs"] / "teacher_dashboard.csv"
    dashboard.to_csv(out_path, index=False)
    print(f"Teacher dashboard saved to {out_path}")
    print(dashboard.head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Student Performance Prediction Agent",
    )
    parser.add_argument("--train", action="store_true", help="Train ML models")
    parser.add_argument("--predict", action="store_true", help="Run demo prediction")
    parser.add_argument("--dashboard", action="store_true", help="Generate teacher dashboard")
    args = parser.parse_args()

    if not any([args.train, args.predict, args.dashboard]):
        parser.print_help()
        print("\nTip: run `python main.py --train --predict` for the full demo.")
        return

    if args.train:
        cmd_train()
    if args.predict:
        cmd_predict()
    if args.dashboard:
        cmd_dashboard()


if __name__ == "__main__":
    main()
