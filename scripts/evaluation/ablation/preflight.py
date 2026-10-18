"""The frozen ablation protocol must run from its original source checkout.

Current production removed HyDE, CRAG and the separate risk-based verifier.
Reusing its old full/disabled profiles would mislabel different implementations.
Run the current backend contract tests to validate current capabilities.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.parse_args()
    parser.error('Historical ablation profiles require their recorded Git checkout; current contracts are tested with pytest backend/app/tests')


if __name__ == '__main__':
    main()
