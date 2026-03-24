"""run_all.py
-----------
Run BS PINN, Heston PINN, and calibration pipelines end-to-end.
Writes a full stdout log to code/docs/results.md.

Calls each pipeline in-process (torch DLLs can't load in Windows subprocesses).

  python run_all.py [--steps N]
"""

import argparse
import io
import sys
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

SCRIPTS = Path(__file__).parent
DOCS    = SCRIPTS.parent / "docs"
DOCS.mkdir(parents=True, exist_ok=True)


def call_pipeline(module, extra_argv=()):
    """Run module.main() with patched sys.argv, capture and print output."""
    old_argv = sys.argv
    sys.argv  = [module.__file__] + list(extra_argv)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            module.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    text = buf.getvalue()
    print(text, end="")
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    args = p.parse_args()
    steps = str(args.steps)

    # Import pipelines (deferred so torch loads once in this process)
    from fys5429 import pipeline as pipe_bs
    from fys5429 import pipeline_heston as pipe_h
    from fys5429 import pipeline_calibrate as pipe_cal

    runs = [
        ("pipeline",           pipe_bs,  ["--steps", steps]),
        ("pipeline_heston",    pipe_h,   ["--steps", steps]),
        ("pipeline_calibrate", pipe_cal, []),
    ]

    outputs = {}
    for name, mod, extra in runs:
        print(f"\n{'='*60}\n  {name}.py\n{'='*60}")
        out = call_pipeline(mod, extra)
        outputs[name] = out

    # Write results.md
    today = date.today().isoformat()
    doc   = [f"# Preliminary Results — {today}\n",
             f"Training steps per PINN: {steps}\n"]
    for name, out in outputs.items():
        doc.append(f"\n## {name}\n")
        doc.append(f"```\n{out.strip()}\n```\n")

    out_path = DOCS / "results.md"
    out_path.write_text("\n".join(doc), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
