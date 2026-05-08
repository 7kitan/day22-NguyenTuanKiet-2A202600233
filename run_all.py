"""
run_all.py — Orchestrator for all 4 steps
==========================================

Run all steps sequentially:
  python run_all.py

Run a specific step:
  python run_all.py --step 1
  python run_all.py --step 2
  python run_all.py --step 3
  python run_all.py --step 4
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_step(step_num: int):
    """Run a specific step."""
    script_map = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py",
    }

    if step_num not in script_map:
        print(f"❌ Invalid step: {step_num}. Choose 1-4.")
        return False

    script = script_map[step_num]
    script_path = Path(script)

    if not script_path.exists():
        print(f"❌ Script not found: {script}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  Running Step {step_num}: {script}")
    print(f"{'=' * 60}\n")

    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            cwd=Path(__file__).parent,
        )
        print(f"\n✅ Step {step_num} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step {step_num} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error running step {step_num}: {e}")
        return False


def run_all_steps():
    """Run all 4 steps sequentially."""
    print("=" * 60)
    print("  Day 22 Lab — All Steps")
    print("=" * 60)

    results = {}
    for step in [1, 2, 3, 4]:
        results[step] = run_step(step)
        if not results[step]:
            print(f"\n⚠️  Step {step} failed. Continuing to next step...")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for step in [1, 2, 3, 4]:
        status = "✅ Pass" if results[step] else "❌ Fail"
        print(f"  Step {step}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All steps completed successfully!")
    else:
        print("\n⚠️  Some steps failed. Review the output above.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Run Day 22 lab steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py              # Run all steps
  python run_all.py --step 1     # Run only step 1
  python run_all.py --step 3     # Run only step 3
        """,
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run a specific step (1-4)",
    )

    args = parser.parse_args()

    if args.step:
        success = run_step(args.step)
        sys.exit(0 if success else 1)
    else:
        success = run_all_steps()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
