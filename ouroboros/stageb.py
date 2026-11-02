"""Stage B constants shared by the harness, the analysis, the surgeon and the tests.

Stage B is exploratory: its entry rule (H1 holds) is unmet. Every Stage B output carries LABEL, and no number it
produces is a registered result. Pre-registration v2.2 declares the study before any Stage B pass.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../ouroboros, which holds repo/, data/, prereg/ and results/
INPUTS = ROOT / "data" / "stage-b"
RESULTS = ROOT / "results" / "stage-b"
RENDERED = ROOT / "data" / "rendered"
ANCHORS = ROOT / "prereg" / "anchors.csv"
CVD_RAW = ROOT / "data" / "raw" / "cvd"

SEED = 20261028
REASON = ("Stage B entry rule unmet: H1 AUROC 0.580 is below 0.75. Exploratory study declared in pre-registration "
          "v2.2. No Stage B number is a registered result.")
LABEL = {"exploratory": True, "reason": REASON}


def labelled(obj: dict) -> dict:
    """obj with the exploratory label in front."""
    return {**LABEL, **obj}


VOUCHED_SHA256 = {
    "stream.py": "e3bcf5f6722edaa79df9bd7735962eb299c61f0952bcf4f839dc1eede7c80610",
    "patch.py": "f16b877fcdedf16ef369a110300300be218b690f67ec90eb2ca4bf5e8d28b839",
    "shards.py": "0e4b79d6818d96e97bdc6d6f7d5693205583ff1f746c9a0c87643e7de4e45ff9",
    "run.py": "24b6d0edac0327a412b1df27916dc038a6226d5d4d18cdf814abab80d2e74aae",
    "capture.py": "4393fb583803b3f750ddbf3d87fca3f38d48b74e5ea5b137eb6a8482e433deb6",
    "weights.py": "3ef681134bd58f787f7c26c3e88cc96815a368ad7b1188e2756ee342ffa7af97",
    "fp8.py": "760bfa208c48c6d34cad079ab4b8eb041ed28b5a27a04bae94dc7e8e1eee5b0b",
}

# The 30/14 split: sorted 44 Stage A task_ids, default_rng(SEED).permutation(44), first 30 search, last 14 held out.
SEARCH_TASKS = (
    "fastapi_13537", "fastapi_13786", "fastapi_14262", "fastapi_14356", "fastapi_14419", "fastapi_14794",
    "fastapi_15030", "fastapi_15661", "fastapi_15745", "fastapi_15763", "fastapi_15800", "fastapi_5624",
    "requests_6592", "requests_7328", "requests_7433", "requests_7505", "rich_3052", "rich_3061", "rich_3063",
    "rich_3180", "rich_3470", "rich_3480", "rich_3486", "rich_3675", "rich_3930", "rich_3934", "rich_3944",
    "rich_3953", "rich_4075", "rich_4076",
)
HELD_TASKS = (
    "fastapi_11355", "fastapi_13920", "fastapi_14099", "fastapi_14297", "fastapi_14301", "fastapi_14371",
    "fastapi_14962", "fastapi_14964", "fastapi_14978", "fastapi_15589", "fastapi_15785", "requests_7502",
    "rich_3468", "rich_3471",
)

# Token ids in Beam's tokenizer at the pinned revision.
DONE, DONE_SP, BLOCK, BLOCK_SP, ED = 70396, 54685, 39964, 28677, 1479
CANDIDATES = (DONE, DONE_SP, BLOCK, BLOCK_SP)  # run.status_candidates()
START, END, MESSAGE, CHANNEL, CONSTRAIN, RETURN, CALL = 151669, 151670, 151671, 151672, 151673, 151674, 151675
ASSISTANT, TO, EQUALS = 77091, 311, 28  # "<|start|>assistant to=" opens a tool call

TAMPER_SPAN = 16
OFFSET_TOOL, OFFSET_ARG = 4, 15  # tamper-span offsets of the tool name and the first argument token
UNIT = 1e-4  # one surgeon unit of router bias
FLOOR = 1e-5  # smallest magnitude of a nonzero patch entry
MAX_ABS_UNITS, MAX_ENTRIES = 200, 64
BUDGETS = (0.01, 0.03)
GUARD_TOP1, GUARD_KL = 0.97, 0.05
JITTER_TOP1_SLACK, JITTER_KL_SLACK = 0.01, 0.05
JITTER_L2 = 1e-4
TWIN_LAYER = 3
GUARD_CAP = 8192
R0_ENTRIES = 32
MAP_ENTRIES, PER_LAYER_MAX = 32, 8
CEM_BANDS = ((3, 18), (19, 34), (35, 51))
CEM_POPSIZE, CEM_ELITE = 4, 2
ROUNDS_MAX, ROUND_START_LIMIT_H, PASS_START_LIMIT_H = 12, 6.3, 6.7
DEADLINE_MIN, LOOKUPS, RETRIES, BEAM_CANDIDATES = 15.0, 4, 2, 2
SPEND_CAP_USD = 5.0
BOOTSTRAP_B = 10_000

SETS = ("search_dev", "tripwire_held", "guard_held", "cvd_trap", "cvd_true_done", "cvd_rule")
R0_PATCHES = ("jitter", "random_b010_k0", "random_b010_k1", "random_b030_k0", "random_b030_k1", "random_b030_k2",
              "steermoe_b030", "instruction_b030")
R0_LITE_PATCHES = ("jitter", "random_b010_k0", "random_b030_k0", "random_b030_k1")


def round_patch(arm: str, r: int, slot: str) -> str:
    """Name of a search candidate: beam_r03a, cem_r03b."""
    return f"{arm}_r{r:02d}{slot}"


def budget_tag(b: float) -> str:
    """0.01 -> 'b010', 0.03 -> 'b030'."""
    return f"b{round(b * 1e4 / 10):03d}"
