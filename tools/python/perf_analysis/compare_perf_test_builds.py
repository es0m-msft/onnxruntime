#!/usr/bin/env python3
"""
Compare performance of multiple onnxruntime_perf_test.exe builds across multiple models.

Usage:
    # Run benchmarks and capture output:
    python compare_perf_test_builds.py \\
        --build "C:/build1/Release" \\
        --build "C:/build2/Release" \\
        --models "C:/models" \\
        --runs 20 \\
        --cooldown 30

    # Parse existing captured output:
    python compare_perf_test_builds.py \\
        --parse-dir ./raw_output \\
        --output results
"""

import argparse
import subprocess
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import sys
import json
import os
from datetime import datetime


def parse_perf_output(output: str, stderr: str, exit_code: int) -> dict:
    """Parse onnxruntime_perf_test.exe output to extract metrics."""
    results = {'exit_code': exit_code}

    if exit_code != 0:
        results['error'] = f"Exit code {exit_code}"
        if stderr:
            results['error'] += f": {stderr[:200]}"
        return results

    for line in output.split('\n'):
        line = line.strip()

        # Try different average formats
        if 'average' in line.lower() and 'ms' in line.lower():
            # Format 1: "Average inference time cost total: 105.452300 ms"
            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        avg_str = parts[1].strip().split()[0]
                        results['average'] = float(avg_str)
                    except (ValueError, IndexError):
                        pass
            # Format 2: "Average: 105.45 ms"
            elif 'Average' in line:
                try:
                    avg_str = line.split('Average')[1].strip().split()[0].replace(':', '')
                    results['average'] = float(avg_str)
                except (ValueError, IndexError):
                    pass

        elif line.startswith('P50 Latency:') or line.startswith('P50:'):
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    p50_str = parts[1].strip().split()[0]
                    results['median'] = float(p50_str) * 1000  # Convert s to ms
                except (ValueError, IndexError):
                    pass

        elif line.startswith('P90 Latency:') or line.startswith('P90:'):
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    p90_str = parts[1].strip().split()[0]
                    results['p90'] = float(p90_str) * 1000  # Convert s to ms
                except (ValueError, IndexError):
                    pass

        elif line.startswith('Min Latency:') or line.startswith('Min:'):
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    min_str = parts[1].strip().split()[0]
                    results['min'] = float(min_str) * 1000  # Convert s to ms
                except (ValueError, IndexError):
                    pass

        elif line.startswith('Max Latency:') or line.startswith('Max:'):
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    max_str = parts[1].strip().split()[0]
                    results['max'] = float(max_str) * 1000  # Convert s to ms
                except (ValueError, IndexError):
                    pass

        elif 'CPU' in line and 'usage' in line:
            try:
                cpu_str = line.split(':')[1].strip().replace('%', '').strip()
                results['cpu'] = int(cpu_str)
            except (ValueError, IndexError):
                pass

    return results


def save_raw_output(output_dir: Path, build_name: str, model_name: str, stdout: str, stderr: str, exit_code: int):
    """Save raw output to file for later parsing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"{build_name}_{model_name}.json"
    data = {
        'stdout': stdout,
        'stderr': stderr,
        'exit_code': exit_code,
        'timestamp': datetime.now().isoformat()
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def run_benchmark(perf_test_path: Path, model_path: Path, runs: int, cooldown: int,
                 output_dir: Optional[Path], build_name: str, model_name: str) -> Dict[str, Any]:
    """Run performance test on a single model."""
    print(f"  Running {perf_test_path.name} on {model_path.name}...")
    print(f"  Cooldown: {cooldown}s...")
    time.sleep(cooldown)

    cmd = [
        str(perf_test_path),
        "-I", str(model_path),
        "-m", "times",
        "-r", str(runs)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Save raw output
        if output_dir:
            save_raw_output(output_dir, build_name, model_name,
                          result.stdout, result.stderr, result.returncode)

        # Parse and return
        parsed = parse_perf_output(result.stdout, result.stderr, result.returncode)

        # Print summary
        if 'error' in parsed:
            print(f"    ERROR: {parsed['error']}")
        elif 'average' in parsed:
            print(f"    Average: {parsed['average']:.2f}ms, CPU: {parsed.get('cpu', 0)}%")
        else:
            print(f"    WARNING: Could not parse average from output")

        return parsed

    except subprocess.TimeoutExpired:
        error_result = {"error": "Timeout after 600s", "exit_code": -1}
        if output_dir:
            save_raw_output(output_dir, build_name, model_name, "", "Timeout", -1)
        print(f"    ERROR: Timeout")
        return error_result

    except Exception as e:
        error_result = {"error": str(e), "exit_code": -1}
        if output_dir:
            save_raw_output(output_dir, build_name, model_name, "", str(e), -1)
        print(f"    ERROR: {str(e)}")
        return error_result


def load_raw_outputs(parse_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """Load and parse raw output files from directory."""
    results = {}

    json_files = list(parse_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No JSON files found in {parse_dir}")
        return results

    print(f"Found {len(json_files)} raw output files")

    for json_file in json_files:
        # Parse filename: BuildName_ModelName.json
        parts = json_file.stem.split('_', 1)
        if len(parts) != 2:
            print(f"WARNING: Skipping {json_file.name} - invalid format")
            continue

        build_name, model_name = parts

        with open(json_file, 'r') as f:
            data = json.load(f)

        parsed = parse_perf_output(data['stdout'], data['stderr'], data['exit_code'])

        if model_name not in results:
            results[model_name] = {}
        results[model_name][build_name] = parsed

    return results


def write_csv(results: Dict[str, Dict[str, Dict]], filename: str):
    """Write results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        if not results:
            return

        first_model = next(iter(results.values()))
        build_names = sorted(first_model.keys())

        header = ["Model"]
        for build in build_names:
            header.extend([
                f"{build}_Average(ms)",
                f"{build}_Median(ms)",
                f"{build}_P90(ms)",
                f"{build}_CPU(%)",
                f"{build}_Min(ms)",
                f"{build}_Max(ms)",
                f"{build}_ExitCode"
            ])
        writer.writerow(header)

        # Data rows
        for model_name in sorted(results.keys()):
            model_results = results[model_name]
            row = [model_name]
            for build in build_names:
                data = model_results.get(build, {})
                if "error" in data:
                    row.extend(["ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", data['exit_code']])
                else:
                    row.extend([
                        f"{data.get('average', 0):.2f}",
                        f"{data.get('median', 0):.2f}",
                        f"{data.get('p90', 0):.2f}",
                        f"{data.get('cpu', 0):.0f}",
                        f"{data.get('min', 0):.2f}",
                        f"{data.get('max', 0):.2f}",
                        f"{data.get('exit_code', 0)}"
                    ])
            writer.writerow(row)


def write_text_report(results: Dict[str, Dict[str, Dict]], filename: str, args):
    """Write human-readable text report."""
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ONNX Runtime Performance Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if hasattr(args, 'runs'):
            f.write(f"Runs per model: {args.runs}\n")
            f.write(f"Cooldown period: {args.cooldown}s\n")

        if hasattr(args, 'build_dirs'):
            f.write("\nBuild Directories:\n")
            for i, build_dir in enumerate(args.build_dirs, 1):
                f.write(f"  Build {i}: {build_dir}\n")
            f.write(f"\nModel Directory: {args.models}\n")
        elif hasattr(args, 'parse_dir'):
            f.write(f"\nParsed from: {args.parse_dir}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Per-model results
        for model_name in sorted(results.keys()):
            model_results = results[model_name]
            f.write(f"Model: {model_name}\n")
            f.write("-" * 80 + "\n\n")

            builds = sorted(model_results.keys())

            # Header
            f.write(f"{'Build':<25} {'Avg(ms)':>10} {'Median(ms)':>12} {'P90(ms)':>10} {'CPU(%)':>8} {'Min(ms)':>10} {'Max(ms)':>10}\n")
            f.write("-" * 95 + "\n")

            # Data rows
            avg_values = []
            for build in builds:
                data = model_results[build]
                if "error" in data or "average" not in data:
                    error_msg = data.get('error', 'Unknown error')
                    f.write(f"{build:<25} ERROR: {error_msg}\n")
                else:
                    f.write(
                        f"{build:<25} "
                        f"{data.get('average', 0):>10.2f} "
                        f"{data.get('median', 0):>12.2f} "
                        f"{data.get('p90', 0):>10.2f} "
                        f"{data.get('cpu', 0):>8.0f} "
                        f"{data.get('min', 0):>10.2f} "
                        f"{data.get('max', 0):>10.2f}\n"
                    )
                    avg_values.append((build, data['average']))

            # Comparison summary
            if len(avg_values) > 1:
                f.write("\n")
                avg_values_sorted = sorted(avg_values, key=lambda x: x[1])
                fastest = avg_values_sorted[0]

                f.write(f"Fastest: {fastest[0]} ({fastest[1]:.2f}ms)\n")
                f.write("Relative Performance:\n")
                for build, avg in avg_values_sorted:
                    speedup = avg / fastest[1]
                    if build == fastest[0]:
                        f.write(f"  {build}: 1.00x (baseline)\n")
                    else:
                        f.write(f"  {build}: {speedup:.2f}x slower ({avg - fastest[1]:.2f}ms slower)\n")

            f.write("\n" + "=" * 80 + "\n\n")

        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Calculate average performance across all models
        build_averages = {}
        for model_name, model_results in results.items():
            for build, data in model_results.items():
                if "error" not in data and "average" in data:
                    if build not in build_averages:
                        build_averages[build] = []
                    build_averages[build].append(data['average'])

        if build_averages:
            f.write("Average Performance Across All Models:\n\n")
            f.write(f"{'Build':<25} {'Avg Latency(ms)':>18} {'Median Latency(ms)':>20}\n")
            f.write("-" * 65 + "\n")

            build_avg_summary = []
            for build in sorted(build_averages.keys()):
                values = build_averages[build]
                avg = sum(values) / len(values)
                median = sorted(values)[len(values) // 2]
                f.write(f"{build:<25} {avg:>18.2f} {median:>20.2f}\n")
                build_avg_summary.append((build, avg))

            if len(build_avg_summary) > 1:
                f.write("\n")
                build_avg_summary_sorted = sorted(build_avg_summary, key=lambda x: x[1])
                fastest = build_avg_summary_sorted[0]

                f.write(f"Overall Fastest: {fastest[0]} ({fastest[1]:.2f}ms avg)\n")
                f.write("Overall Relative Performance:\n")
                for build, avg in build_avg_summary_sorted:
                    speedup = avg / fastest[1]
                    if build == fastest[0]:
                        f.write(f"  {build}: 1.00x (baseline)\n")
                    else:
                        f.write(f"  {build}: {speedup:.2f}x slower\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance of multiple onnxruntime_perf_test.exe builds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run benchmarks:
  python %(prog)s \\
    --build C:/build1/Release/Release \\
    --build C:/build2/Release/Release \\
    --models C:/models \\
    --runs 20 \\
    --cooldown 30 \\
    --raw-output ./raw_output

  # Parse existing raw output:
  python %(prog)s \\
    --parse-dir ./raw_output \\
    --output results
        """
    )

    # Mutually exclusive: either run benchmarks or parse existing
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--build",
        action="append",
        dest="build_dirs",
        help="Directory containing onnxruntime_perf_test.exe (can be specified multiple times)"
    )
    mode_group.add_argument(
        "--parse-dir",
        type=str,
        help="Directory containing raw output JSON files to parse"
    )

    # Benchmark mode arguments
    parser.add_argument(
        "--models",
        help="Directory containing .onnx model files (required for --build mode)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of runs per model (default: 20)"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Cooldown period in seconds between executions (default: 30)"
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default="./raw_output",
        help="Directory to save raw output files (default: ./raw_output)"
    )

    # Common arguments
    parser.add_argument(
        "--output",
        default="perf_comparison",
        help="Output filename prefix (default: perf_comparison)"
    )

    args = parser.parse_args()

    # Parse mode
    if args.parse_dir:
        print("=== PARSE MODE ===")
        parse_dir = Path(args.parse_dir).resolve()
        if not parse_dir.exists():
            print(f"ERROR: Parse directory does not exist: {parse_dir}")
            sys.exit(1)

        results = load_raw_outputs(parse_dir)

    # Benchmark mode
    else:
        if not args.models:
            print("ERROR: --models is required when using --build")
            sys.exit(1)

        print("=== BENCHMARK MODE ===")

        # Validate build directories
        build_dirs = []
        for build_dir in args.build_dirs:
            build_path = Path(build_dir).resolve()
            if not build_path.exists():
                print(f"ERROR: Build directory does not exist: {build_path}")
                sys.exit(1)

            perf_test = build_path / "onnxruntime_perf_test.exe"
            if not perf_test.exists():
                print(f"ERROR: onnxruntime_perf_test.exe not found in: {build_path}")
                sys.exit(1)

            build_dirs.append(build_path)

        # Validate model directory
        model_dir = Path(args.models).resolve()
        if not model_dir.exists():
            print(f"ERROR: Model directory does not exist: {model_dir}")
            sys.exit(1)

        # Find all .onnx files
        model_files = sorted(model_dir.glob("*.onnx"))
        if not model_files:
            print(f"ERROR: No .onnx files found in: {model_dir}")
            sys.exit(1)

        # Create raw output directory
        raw_output_dir = Path(args.raw_output)
        raw_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(build_dirs)} build(s) and {len(model_files)} model(s)")
        print(f"Runs per model: {args.runs}")
        print(f"Cooldown between runs: {args.cooldown}s")
        print(f"Raw output directory: {raw_output_dir}")
        print()

        # Run benchmarks
        results = {}
        for model_path in model_files:
            model_name = model_path.stem
            print(f"Benchmarking model: {model_name}")
            results[model_name] = {}

            for i, build_dir in enumerate(build_dirs, 1):
                build_name = f"Build{i}_{build_dir.parent.name}"
                perf_test = build_dir / "onnxruntime_perf_test.exe"

                result = run_benchmark(perf_test, model_path, args.runs, args.cooldown,
                                     raw_output_dir, build_name, model_name)
                results[model_name][build_name] = result

            print()

    # Generate outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{args.output}_{timestamp}.csv"
    txt_file = f"{args.output}_{timestamp}.txt"

    # Write CSV
    write_csv(results, csv_file)
    print(f"\nCSV output written to: {csv_file}")

    # Write text report
    write_text_report(results, txt_file, args)
    print(f"Text report written to: {txt_file}")

    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()
