"""Explicit preflight, non-benchmark provider smoke, or official collection."""
import argparse
import json
from benchmark.orchestration.phase2_preflight_v2 import run_preflight
from benchmark.orchestration.phase2_runner_v2 import run_phase2
from benchmark.providers.transformers_phase2 import OFFICIAL_MODELS
from benchmark.orchestration.provider_smoke import run_provider_smoke, run_provider_context_smoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=OFFICIAL_MODELS, required=True)
    parser.add_argument('--output-root', default='.')
    parser.add_argument('--mode', choices=('environment', 'preflight', 'provider-smoke', 'provider-context-smoke', 'run'), default='environment')
    parser.add_argument('--context-smoke-input-tokens', type=int, default=8700)
    args = parser.parse_args()
    if args.mode == 'provider-context-smoke':
        result = run_provider_context_smoke(args.model, args.output_root, input_tokens=args.context_smoke_input_tokens)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result['status'] == 'PASS' else 1
    if args.mode == 'provider-smoke':
        result = run_provider_smoke(args.model, args.output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result['status'] == 'PASS' else 1
    if args.mode != 'run':
        result = run_preflight(args.model, args.output_root, inspect_model=args.mode == 'preflight')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['status'] == 'PASS' else 1
    result = run_phase2(args.model, args.output_root)
    print(json.dumps({k: str(v) for k, v in result.items()}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
