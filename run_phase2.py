"""Explicit environment preflight, Colab readiness check, or official collection."""
import argparse
import json
from benchmark.orchestration.phase2_preflight import run_preflight
from benchmark.orchestration.phase2_runner import run_phase2
from benchmark.providers.transformers_phase2 import OFFICIAL_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=OFFICIAL_MODELS, required=True)
    parser.add_argument('--output-root', default='.')
    parser.add_argument('--mode', choices=('environment', 'preflight', 'run'), default='environment')
    args = parser.parse_args()
    if args.mode != 'run':
        result = run_preflight(args.model, args.output_root, inspect_model=args.mode == 'preflight')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['status'] == 'PASS' else 1
    result = run_phase2(args.model, args.output_root)
    print(json.dumps({k: str(v) for k, v in result.items()}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
