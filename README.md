# Ouroboros

Beam's router as polygraph and scalpel.

Beam-1.0-501B-A23B sends every token through 8 of its 256 experts in each of its 49 MoE layers. Ouroboros records those choices as Beam reads back its own agent work. It asks one question. When Beam is about to tamper with a test, does its routing show it 500 tokens early?

Stage A is the pre-registered test of that question. Stage B is exploratory. It asks Beam to write edits to its own router selection bias. A third study, also exploratory, asks Beam to read the same windows and answer the question in words.

This repository holds the code for all three: the renderer, the layer-streaming harness for one A100 80GB, the Stage A analysis, the Stage B scorer, the surgeon controller and the reader. A CPU test suite runs all of it on a tiny random Beam.

## Links

- Pre-registration: [rairo/ouroboros-beam-prereg](https://huggingface.co/datasets/rairo/ouroboros-beam-prereg). Stage B and the reader are declared in [STAGE_B_EXPLORATORY.md](https://huggingface.co/datasets/rairo/ouroboros-beam-prereg/blob/main/STAGE_B_EXPLORATORY.md), version 2.2 with amendments 2.2.1, 2.2.2 and 2.2.3.
- Results: [rairo/ouroboros-beam](https://huggingface.co/datasets/rairo/ouroboros-beam)
- Model: [reflection/Beam-1.0-501B-A23B-FP8](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8) at revision `7449ff0f`

## Registered results

| Hypothesis | Result | 95% CI | Bar | Verdict |
|---|---|---|---|---|
| H1: routing warns of tampering 500+ tokens ahead, AUROC, n 326 | 0.580 | [0.523, 0.634] | 0.75 | fail |
| H2: routing beats a bag-of-tokens probe on the same windows | +0.089 | [0.006, 0.173] | +0.05, CI above 0 | pass |
| H3: the H1 probe moved to false DONE reports, AUROC, n 95 | 0.408 | [0.281, 0.622] | 0.65 | fail |

The Stage B entry rule required H1. Stage B therefore runs as an exploratory study, read against the registered bars as yardsticks. An exploratory follow-up traces most of the routing signal to the honesty instruction in Tripwire's impossible_honest condition. The results dataset holds every number, with each exploratory one labelled.

## How Beam was used

Beam plays three roles: subject, surgeon and reader.

**As the subject.** The full model ran in FP8 over 713 transcripts of its own agent work from Tripwire and Claimed vs Done, 11,117,080 tokens in all. Each transcript went through Beam's own chat template. The harness recorded the 8 chosen experts in each of the 49 MoE layers for every token.

**As the surgeon.** Boot 1 had yet to start at the declared cutoff, 2026-11-02 04:59 UTC, so Stage B runs on the zero-shot path that pre-registration v2.2 sets for that case. Beam runs through the Hugging Face router on Baseten at reasoning effort high. At each of two budgets, 0.01 and 0.03, one conversation gives it maps of its own routing, a dose table and a footprint tool. It makes up to 4 lookups and submits 3 router-bias patches by tool call, before any score exists. One A100 boot scores its lead patch at the budget R0-lite sets, with its second beside it, on the FP8 weights. Scoring is teacher-forced on Beam's own earlier tokens, on held-out Tripwire tasks and on Claimed vs Done, beside SteerMoE, an instruction patch and random patches. The 12-round search, with a blind optimiser beside Beam, stays in the code as declared, for a later run.

**As the reader.** Beam reads the 681 windows Qwen3-8B read in Stage A, with the same question, one user turn each, at reasoning effort low and temperature 0. Its score is the log-odds of YES at its answer token, from the route's log-probabilities. The router reads Beam's chat and tool-call markup inside a user message as Beam's own control tokens. Amendments 2.2.2 and 2.2.3 put a zero-width space inside every added token of Beam's tokenizer, and Beam's tokenizer then finds zero added-token ids across the 681 windows. The declaration puts a 20-window pilot ahead of the full set, behind a stated gate. The pilot missed the gate: 4 windows stayed unscored against a limit of 1, and 10 of 24 replies held tool-call markup in their reasoning against a limit of 2. The reader stops there. The pilot is the result, reported as counts and replies.

Read by eye, 10 pilot replies carry on the transcript as the coding agent and 10 weigh the question. 2 of the 16 scored windows read YES. Agent-voice replies that answer score weak, from -5.1 to -1.5. The rest spread from -18.6 to +13.8. Two earlier attempts ship beside the pilot: 7 windows under version 2.2, where the router read the markup as control tokens, and a 20-window pilot under 2.2.2, where two tool-call tags survived the escape. The pilot's request logs keep each reply's message, usage and finish reason whole, and trim each per-token log-probability list to the answer region. The declaration logs every response in full, so this trim is a stated deviation. Reader spend across the three attempts: $0.044 over 58 requests.

## Layout

```
ouroboros/
  render/      transcripts to Beam token ids, labels and anchors
  harness/     layer-streaming forward, FP8 weight reads, capture, patches, Stage B scoring
  analysis/    Stage A probes, controls, monitors and report; Stage B inputs, maps and readouts; the reader
  surgeon/     the surgeon: prompts, checks, footprint simulator, file handoff
  stageb.py    Stage B constants: task split, budgets, guard, exploratory label
tests/         CPU suite on a tiny random Beam in the real FP8 layout
vendor/beam/   Beam's config, tokenizer, chat template and weight index
```

| Module | Role |
|---|---|
| `render/build.py` | Renders every Tripwire episode and Claimed vs Done trap run to parquet, with anchors and the token gate |
| `render/template.py` | Beam's chat template and tokenizer, applied the way the serving route applied them |
| `render/sequence.py` | One episode to one token sequence, with message and tool-call boundaries |
| `render/tripwire.py` | Tripwire requests, labels and the anchors f (fork) and d (tamper) |
| `render/cvd.py` | Claimed vs Done requests, labels and the status slot s |
| `render/tamper.py` | Tripwire's tamper detector, reimplemented to name the calls it counts |
| `render/schemas.py` | Tool schemas exactly as each harness sent them |
| `render/anchors.py`, `render/paths.py` | Anchor helpers and data paths |
| `harness/weights.py` | Per-layer weight reads from a local directory or Cloud Storage |
| `harness/fp8.py` | The per-channel FP8 convention, to and from BF16 |
| `harness/stream.py` | Layer-major streaming forward over packed episodes |
| `harness/shards.py` | Episodes packed into shards under a host-RAM budget |
| `harness/capture.py` | Route ids, route scores, window rows and target log-probs per shard |
| `harness/run.py` | The capture command, resumable per shard |
| `harness/gpu_selftest.py` | The registered harness gate |
| `harness/patch.py` | Router-bias patches and layer-prefix reuse |
| `harness/multi.py`, `harness/score.py` | One weight stream that scores the stock model and many patches |
| `harness/bank.py` | Patch banks, each patch with its provenance |
| `harness/sets.py` | Stage B episode sets and target spans |
| `harness/round.py` | Stage B on the GPU machine: the search rounds, or G1a, R0-lite and the zero-shot picks |
| `harness/tinyckpt.py` | A tiny random Beam checkpoint in the real FP8 format |
| `analysis/stage_a.py` | Stage A, one resumable step at a time: features, probes, controls, keyword, qwen, report |
| `analysis/data.py`, `analysis/features.py` | Inputs, analysis windows and window features |
| `analysis/populations.py` | Populations, the registered count checks and the task folds |
| `analysis/probe.py` | The probe, nested task-grouped cross-validation and the cluster bootstrap |
| `analysis/controls.py` | Shuffled labels, random expert sets, single layers, effort strata |
| `analysis/monitors.py` | Tripwire's keyword monitor, and Qwen3-8B asked one question per window |
| `analysis/report.py` | `oof.csv`, `results.json` and `report.md` |
| `analysis/synthetic.py` | A small synthetic Stage A world in the real file layout |
| `analysis/stage_b.py` | The Stage B task split and scoring sets |
| `analysis/stage_b_maps.py` | Expert maps, the dose table and patch banks, from search tasks only; the boot bank from the picks |
| `analysis/stage_b_metrics.py` | The search objective, the guard and the held-out readouts |
| `analysis/beam_reader.py` | Beam as reader: the escape, the tokenizer gate, the pilot and its gate, the answer rule, request logs and the spend ledger |
| `surgeon/controller.py` | One Beam conversation per search round, or per budget on the zero-shot path |
| `surgeon/validate.py` | Checks on every patch and lookup Beam submits |
| `surgeon/footprint.py` | A route-change simulator for patches on captured scores |
| `surgeon/handoff.py` | The file handoff between the controller and the round loop |

Every file the pre-registration hashes ships here byte for byte. A handful of comments, test strings and default paths were swapped for neutral wording and the published layout.

## Run the tests on CPU

Python 3.12.

```
python -m venv .venv
source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
python -m pytest
```

On Windows, activate with `.venv\Scripts\activate`. Transformers comes from a pinned commit on main, where `transformers.models.beam` lives.

The suite builds a tiny random Beam in the real FP8 layout (`harness/tinyckpt.py`). It keeps the tensor names, dtypes, per-channel scales, ignore list and shard pattern of the full checkpoint. Streaming, capture, patches, multi-patch scoring, the Stage B round loop and the readouts all run on it. The streaming tests check the harness against transformers' own `BeamForCausalLM` holding the same weights.

Some tests read real data and skip when it is absent:

| Tests | Needs |
|---|---|
| Real episodes, anchors and rendered parquets | The data directories below |
| FP8 convention on real Beam tensors | `OUROBOROS_NET_TESTS=1`, for HTTP range reads from the Hub |
| Qwen3-8B monitor on the real tokenizer | `OUROBOROS_QWEN_TOKENIZER` set to a Qwen3-8B snapshot directory |
| Stage B sets and maps on a built inputs tree | `STAGEB_INPUTS` set to that tree |
| The reader's records against the Qwen3-8B run | The Stage A release in `results/stage-a/` |

## Data

The code reads data from directories beside the checkout. `render/paths.py` and `stageb.py` resolve them from the package location.

```
work/
  ouroboros/            this repository
  data/raw/tripwire/    episodes.parquet and variants.parquet from Tripwire
  data/raw/cvd/         a snapshot of Claimed vs Done: runs.csv, runs/, prompts/, tasks/
  data/rendered/        written by the renderer
  data/stage-b/         Stage B inputs
  prereg/anchors.csv    from the pre-registration repo
  results/stage-a/      the Stage A release, as the results dataset's stage-a/ folder
  results/stage-b/      Stage B outputs
  results/reader/       reader outputs
```

Stage A used Tripwire at revision `e575c83` and Claimed vs Done at `9e13608`. Render both with:

```
python -I -m ouroboros.render.build
```

## The A100 streaming harness

The FP8 checkpoint weighs 503.3 GB. The harness streams it through one A100 80GB, one decoder layer at a time. Layers run in the outer loop and tokens in the inner loop.

1. **Pack.** `shards.py` packs rendered episodes into shards under a host-RAM budget. Episodes stay whole.
2. **Read.** `weights.py` parses the safetensors index once and plans the byte spans each layer needs. It reads them as parallel ranged requests, from a local directory or a Cloud Storage prefix, into page-locked host buffers. Two buffers alternate, so the next layer loads while the current one runs.
3. **Dequantize.** `fp8.py` turns per-channel FP8 weights into BF16 on the GPU.
4. **Run.** `stream.py` builds each layer as a transformers `BeamDecoderLayer`, runs it over every token of the shard in chunks of whole episodes, then frees it. Hidden states wait in host memory between layers. Packed position ids and `cu_seqlens` keep attention inside each episode, through flash-attention varlen on CUDA and SDPA elsewhere.
5. **Record.** `capture.py` writes the 8 chosen expert ids and their scores for every token and MoE layer. Inside the analysis windows it adds all 256 scores and hidden states at five tap layers. At target tokens it adds next-token NLL and log-probs. A shard lands in a `.partial` directory and takes its final name once whole.
6. **Resume.** `run.py` skips finished shards and redoes unfinished ones. It can mirror each finished shard to a sync location, since a preempted VM loses its local disk. Each shard records the settings that shape it, and a rerun with other settings stops.

The registered gate, `gpu_selftest.py`, runs before any capture. Each streamed layer must match transformers' own layer within BF16 tolerance. Teacher-forced NLL on Beam's own sampled tokens must stay at or under 1.0 nat per token, with argmax agreement of 70% or more. Two runs of one shard must agree on the top-8 set at 99.9% of token-layers or more. On the A100, the streamed model scored 0.370 nats per token over 57,462 sampled tokens, with 87.2% argmax agreement, and a rerun reproduced every top-8 set.

The Stage A capture ran 2.0 h across 4 shards, at about 1,650 tokens per second, and wrote 68.76 GB.

Stage B reuses the same stream. A patch is a 49 x 256 FP32 delta added to each MoE layer's selection bias, `e_score_correction_bias`. The bias picks the experts, and the mixing weights come from the unbiased scores. `multi.py` and `score.py` score the stock model and many patches in one pass over the weights. Each patch shares the stock trunk up to its earliest patched layer and forks there. `round.py` runs the search rounds on the GPU machine. `surgeon/controller.py` runs Beam through the Hugging Face router on the laptop. The two sides meet at a file handoff, so the API token stays on the laptop.

## Stage B on the zero-shot path

Every file these steps write carries the exploratory label.

1. **Write.** On the laptop, `python -m ouroboros.surgeon.controller zero-shot --inputs DIR --out beam_zero_shot.safetensors --allow-paid` runs one conversation per budget. `validate.py` checks every lookup and submission. Each budget's footprint cap is the largest simulated footprint among the R0 random patches at that budget. The six patches land in one bank with its sha256. The controller sends `max_tokens` alone, since the Baseten route refuses a body that carries both output caps (amendment 2.2.1).
2. **Gate.** On the GPU machine, `python -m ouroboros.harness.round r0-lite ... --zero-shot-bank FILE` runs G1a (`run.py` from the vouched tree on `search_dev`), R0-lite (stock, twin, jitter, random at 0.01, random at 0.03 x 2) and the equivalence gate. R0-lite sets the budget B and the floors. Beam's lead patch at B takes Beam's slot. Its second takes the slot the blind optimiser holds in the search design.
3. **Build.** `python -m ouroboros.analysis.stage_b_maps boot2-bank --picks FILE --picks-bank FILE --zero-shot FILE` builds the boot bank and the E1 and E2 variants files.
4. **Score.** `score.py` runs E1 on Claimed vs Done (trap slots, true-DONE slots, the rule re-render) and E2 on held-out Tripwire tamper spans and the guard set.
5. **Read.** `python -m ouroboros.analysis.stage_b_metrics readouts --e1 DIR --e2 DIR --picks FILE --out FILE` computes every yardstick with its cluster-bootstrap interval.

## The reader

```
python -m ouroboros.analysis.beam_reader gate
python -m ouroboros.analysis.beam_reader prereg
python -m ouroboros.analysis.beam_reader run --allow-paid --limit 20
python -m ouroboros.analysis.beam_reader pilot
```

`gate` encodes every escaped window and every user turn with Beam's own tokenizer and stops on any added-token id. `prereg` checks that the head of the pre-registration repo holds the declaration and amendments 2.2.2 and 2.2.3, and records the commit. `run --limit 20` sends the pilot: 6 requests at a time, a $3 cap over every attempt, the token from the `huggingface_hub` login, and the auth header stripped from every log. `pilot` writes `pilot_gate.json` and, when the gate fails, `PILOT.md`. A full `run` needs a passing `pilot_gate.json`, and `readouts` follows it. This study's pilot missed its gate, so the reader ends at `pilot`.

The reader reads the Stage A release (`features/texts_*.jsonl`, `monitors/qwen_*`), `prereg/anchors.csv`, the rendered parquets and `vendor/beam/tokenizer.json`. Each path takes a flag.

## Credits

- [Tripwire](https://huggingface.co/datasets/dhairyapandya/beam-tripwire-episodes), by Dhairya Pandya, Apache-2.0. Beam's episodes on 44 rigged bug-fix tasks, with the tamper detector and keyword monitor this code reimplements.
- [Claimed vs Done](https://huggingface.co/datasets/BrasaStudios/claimed-vs-done), by Brasa Studios, Apache-2.0. Graded status reports from coding agents, used for the H3 transfer test and the Stage B readouts.
- [Inside Beam](https://huggingface.co/spaces/rycerzes/inside-beam), by Swapnil Dutta (rycerzes). Prior expert steering of Beam through the same selection bias.
- [SteerMoE](https://arxiv.org/abs/2509.09660), Fayyaz et al., 2025. Experts found by paired activation frequency, scored here as a Stage B comparison arm.
- [Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8), by Reflection AI, Apache-2.0. The files in `vendor/beam/` come from its FP8 repository.

## Licence

Apache-2.0. Copyright 2026 Ouroboros contributors. `NOTICE` lists the work this code builds on.
