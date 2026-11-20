# Agentic V2X paper experiment reproduction

This is a small, standalone repository for reproducing the Adaptive LLM V2X
scheduler experiments. It contains the study runner and analysis code, the three
scenario entry points, the mobility traces used by the runs, and a focused patch
for two shared files in the external ns-3 checkout.

It does **not** contain an ns-3 source tree, model weights, or experiment result
files. Run outputs are written locally under `results/` and ignored by Git.

## Experiments covered

1. **Original Adaptive LLM cohort:** 100 simulated seconds for each density
   (20, 25, and 30 vehicles) and seed (1–7), for 21 runs total. The local
   `qwen2.5:7b` model proposes policy updates; the xApp validates and applies
   them.
2. **Refresh-disabled ablation:** the same 21 density/seed pairs, starting
   from the validated heuristic policy and suppressing periodic LLM requests.
   The control loop, telemetry, safety shield, rollback checks, and scheduler
   weight writes remain enabled. Run this after the original cohort because
   the analysis pairs each ablation run against its matching original run.

The analysis scripts retain both critical DC-PRR definitions and mark the
original cohort contrasts as exploratory. Read each generated `results.md` and
run manifest before using numbers in a paper.

## Repository contents

- `scripts/` and `src/llm_v2x_scheduler/`: study orchestration, xApp, validation,
  verification, and analysis code.
- `ns3/`: the three small scenario entry points built by ns-3.
- `integration/ns3-adaptive-study.patch`: the required changes to the external
  O-RAN scenario helper and shared-memory Python interface.
- `mobility_inputs/`: the three input mobility traces used in the experiments.
- `integration/dependency-pins.txt`: source revisions, model digest, and trace
  checksums for the recorded experiment environment.

## Prerequisites

Use Linux and a compatible ns-3 checkout with the NR and ns3-ai components.
The recorded revisions are in `integration/dependency-pins.txt`. The experiment
repository must be placed at exactly:

```text
<ns-3-root>/scratch/llm_v2x_scheduler
```

The scripts locate the ns-3 root relative to that path. The ns-3 tree is a
separate dependency and is intentionally not copied into this repository.

You will also need:

- a C++ build toolchain and CMake supported by the pinned ns-3 revision;
- Boost.Program_options development headers (for `contrib/ai`);
- Python 3.10 or newer, using the same Python executable for the xApp agent;
- Python package `PyYAML`;
- Ollama running locally at `http://127.0.0.1:11434`, with `qwen2.5:7b`
  installed for the original cohort. The recorded model digest is listed in
  `integration/dependency-pins.txt`.

The mobility traces are included, so SUMO is not needed to run these two
experiments.

## Prepare the external ns-3 checkout

Start from ns-3, NR, and ns3-ai checkouts at the revisions listed in
`integration/dependency-pins.txt`. Place this repository at
`$NS3_ROOT/scratch/llm_v2x_scheduler`, then from the ns-3 root apply the small
integration patch and install the included traces at the paths expected by the
scenarios:

```bash
export NS3_ROOT=/path/to/ns-3-dev
export REPRO="$NS3_ROOT/scratch/llm_v2x_scheduler"

cd "$NS3_ROOT"
git apply --check "$REPRO/integration/ns3-adaptive-study.patch"
git apply "$REPRO/integration/ns3-adaptive-study.patch"

mkdir -p "$NS3_ROOT/scratch/mobility_1"
cp "$REPRO"/mobility_inputs/ns2manhattanmobility{20,25,30}.tcl \
  "$NS3_ROOT/scratch/mobility_1/"
```

Install Python and build the optimized scenario binaries:

```bash
python3 -m pip install PyYAML
ollama pull qwen2.5:7b

cd "$NS3_ROOT"
./ns3 configure --build-profile=optimized
cmake --build cmake-cache --target \
  scratch_llm_v2x_scheduler_scenario_llm_v2x_s20 \
  scratch_llm_v2x_scheduler_scenario_llm_v2x_s25 \
  scratch_llm_v2x_scheduler_scenario_llm_v2x_s30 -j 3
```

The run manifest records the actual binary, trace, source, and model hashes.
For a byte-for-byte environment match, compare them with
`integration/dependency-pins.txt` and the metadata produced by your run.

## Run the experiments

Run the original 21-case cohort first. This serial command resumes successful
trials from their manifests if interrupted:

```bash
cd "$REPRO"
python3 -B scripts/run_adaptive_study.py \
  --phase original --jobs 1 \
  --output results/adaptive_revision_v2_adaptive_prompt_fix
```

The runs take substantial wall time because each 100-second simulation runs
with a local language model. The output is written beneath
`results/adaptive_revision_v2_adaptive_prompt_fix/original/`. After completion,
create the seed-level cohort analysis with:

```bash
python3 -B scripts/analyze_original_cohort.py
```

Then run the matched refresh-disabled ablation and its paired analysis:

```bash
python3 -B scripts/run_refresh_disabled_ablation.py \
  --output results/refresh_disabled_ablation
```

The ablation runner verifies each case, resumes already verified cases, and
writes the paired summary to `results/refresh_disabled_ablation/results.md`.
The run-level CSV, seed-block CSV, guard activity, and machine-readable summary
are written beside it. The cohort analysis files are written under
`results/adaptive_revision_v2_adaptive_prompt_fix/`.

## Reproducibility notes

- The simulator uses ns-3 RNG seed `1`; each matrix seed is the ns-3 RNG run
  number. The xApp receives 980 control windows in a successful 100-second run.
- The Adaptive LLM cohort uses one model digest and fixed generation options;
  the runner records these in `ollama_model.json` and each run manifest.
- The ablation is paired by density and run index. Do not change source, model,
  trace, scenario settings, or output directory between cases in one campaign.
- `results/` is intentionally excluded from version control. Keep generated
  manifests and outputs with any analysis release or archive them separately.
