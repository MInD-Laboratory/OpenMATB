#!/usr/bin/env python3

from __future__ import annotations

import configparser
import json
import re
import shutil
import subprocess
import sys
import ast
from statistics import mean
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.ini"
SCENARIOS_DIR = ROOT / "includes" / "scenarios"
SESSIONS_DIR = ROOT / "sessions"
CAPACITY_TARGET_SCORE = 0.75
CAPACITY_ALPHA = 0.80
CALIBRATION_SEGMENT_SECONDS = 120
M_MIN = 0.60
M_MAX = 1.20

PERMUTATIONS = [
    ("L", "M", "H"),
    ("L", "H", "M"),
    ("M", "L", "H"),
    ("M", "H", "L"),
    ("H", "L", "M"),
    ("H", "M", "L"),
]


def parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def load_openmatb_config() -> tuple[str, int, str, str, str, bool]:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    if "Openmatb" not in config:
        raise RuntimeError("[Openmatb] section missing in config.ini")

    section = config["Openmatb"]
    experiment = section.get("experiment", "A").strip().upper()

    participant_raw = section.get("participant_id", "401").strip()
    try:
        participant_id = int(participant_raw)
    except ValueError as exc:
        raise RuntimeError(f"participant_id must be an integer, got: {participant_raw}") from exc

    current_scenario_path = section.get("scenario_path", "").strip()
    stream_b_order = section.get("stream_b_order", "auto").strip().lower()
    stream_c_condition = section.get("stream_c_condition", "auto").strip().lower()
    record_face = parse_bool(section.get("record_face", "true"), default=True)
    return experiment, participant_id, current_scenario_path, stream_b_order, stream_c_condition, record_face


def parse_hms_to_seconds(value: str) -> int | None:
    match = re.match(r"^(\d+):(\d{2}):(\d{2})$", value.strip())
    if match is None:
        return None
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s)


def format_seconds_to_hms(value: int) -> str:
    hh = value // 3600
    rem = value % 3600
    mm = rem // 60
    ss = rem % 60
    return f"{hh}:{mm:02d}:{ss:02d}"


def build_runtime_facecamera_scenario(rel_scenario_path: str) -> str:
    src_path = SCENARIOS_DIR / rel_scenario_path
    lines = src_path.read_text(encoding="utf-8").splitlines()

    if any(";facecamera;" in line for line in lines):
        return rel_scenario_path

    max_sec = 0
    has_lsl = False
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0 or stripped.startswith("#"):
            continue

        if ";labstreaminglayer;" in stripped:
            has_lsl = True

        pieces = stripped.split(";", 1)
        if len(pieces) < 2:
            continue

        sec = parse_hms_to_seconds(pieces[0])
        if sec is not None and sec > max_sec:
            max_sec = sec

    stop_ts = format_seconds_to_hms(max_sec)

    injected_lines = list(lines)
    injected_lines.append("")
    injected_lines.append("# Runtime injected face camera controls")
    injected_lines.append("0:00:00;facecamera;start")
    if has_lsl:
        injected_lines.append("0:00:00;labstreaminglayer;marker;FACECAM_START")
    if has_lsl:
        injected_lines.append(f"{stop_ts};labstreaminglayer;marker;FACECAM_STOP")
    injected_lines.append(f"{stop_ts};facecamera;stop")

    runtime_dir = SCENARIOS_DIR / "_runtime_facecamera"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime_name = rel_scenario_path.replace("/", "__").replace(".txt", "") + "__facecamera.txt"
    runtime_rel_path = f"_runtime_facecamera/{runtime_name}"
    runtime_path = SCENARIOS_DIR / runtime_rel_path
    runtime_path.write_text("\n".join(injected_lines) + "\n", encoding="utf-8")
    return runtime_rel_path


def set_openmatb_value(key: str, value: str) -> None:
    lines = CONFIG_PATH.read_text(encoding="utf-8").splitlines()

    in_openmatb = False
    section_start = None
    section_end = None
    key_index = None

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            if in_openmatb:
                section_end = idx
                break
            in_openmatb = stripped.lower() == "[openmatb]"
            if in_openmatb:
                section_start = idx
            continue

        if in_openmatb and stripped.startswith(f"{key}="):
            key_index = idx
            break

    if section_start is None:
        raise RuntimeError("[Openmatb] section missing in config.ini")

    if key_index is not None:
        lines[key_index] = f"{key}={value}"
    else:
        insert_at = section_end if section_end is not None else len(lines)
        lines.insert(insert_at, f"{key}={value}")

    CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_exp_a_block_sequence(participant_id: int) -> list[tuple[str, str]]:
    offset = participant_id - 401
    perm_index = offset % 6
    block_order = PERMUTATIONS[perm_index]

    blocks: list[tuple[str, str]] = [
        ("Instruction Subtasks", "expA/expA_instructions_subtasks.txt"),
        ("Instruction Combined (Low)", "expA/expA_instructions_combined_L.txt"),
    ]

    for i, key in enumerate(block_order, start=1):
        blocks.append((f"Baseline Block {i} ({key})", f"expA/participant_{participant_id}/expA_pre_{i}_{key}.txt"))

    for i, key in enumerate(block_order, start=1):
        blocks.append((f"Experimental Block {i} ({key})", f"expA/participant_{participant_id}/expA_main_{i}_{key}.txt"))

    return blocks


def build_exp_b_block_sequence(participant_id: int, stream_b_order: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = [
        ("Instruction Subtasks", "expB/expB_instructions_subtasks.txt"),
        ("Instruction Combined (Low)", "expB/expB_instructions_combined_L.txt"),
        ("Practice Block (10 min, m=0.85)", f"expB/participant_{participant_id}/expB_practice_M_10min.txt"),
        ("Integrated Calibration (8 min)", f"expB/participant_{participant_id}/expB_calibration_integrated_8min.txt"),
        ("Baseline Anchor (2 min, m=0.60)", f"expB/participant_{participant_id}/expB_baseline_anchor_2min.txt"),
    ]

    if stream_b_order not in {"auto", "low-high", "high-low"}:
        raise ValueError("stream_b_order must be one of: auto, low-high, high-low")

    if stream_b_order == "auto":
        low_first = (participant_id % 2 == 0)
    else:
        low_first = stream_b_order == "low-high"

    if low_first:
        blocks.extend(
            [
                ("Experimental Block 1 (Low, 25 min)", f"expB/participant_{participant_id}/expB_experimental_low_25min.txt"),
                ("Experimental Block 2 (High, 25 min)", f"expB/participant_{participant_id}/expB_experimental_high_25min.txt"),
            ]
        )
    else:
        blocks.extend(
            [
                ("Experimental Block 1 (High, 25 min)", f"expB/participant_{participant_id}/expB_experimental_high_25min.txt"),
                ("Experimental Block 2 (Low, 25 min)", f"expB/participant_{participant_id}/expB_experimental_low_25min.txt"),
            ]
        )
    return blocks


def build_exp_c_block_sequence(participant_id: int, stream_c_condition: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = [
        ("Instruction Subtasks", "expC/expC_instructions_subtasks.txt"),
        ("Instruction Combined (Low)", "expC/expC_instructions_combined_L.txt"),
        ("Practice Block (10 min, m=0.85)", f"expC/participant_{participant_id}/expC_practice_M_10min.txt"),
        ("Integrated Calibration (8 min)", f"expC/participant_{participant_id}/expC_calibration_integrated_8min.txt"),
        ("Baseline Anchor (2 min, m=0.60)", f"expC/participant_{participant_id}/expC_baseline_anchor_2min.txt"),
    ]

    if stream_c_condition not in {"auto", "low", "high"}:
        raise ValueError("stream_c_condition must be one of: auto, low, high")

    if stream_c_condition == "auto":
        condition = "low" if participant_id % 2 == 0 else "high"
    else:
        condition = stream_c_condition

    if condition == "low":
        blocks.append(("Experimental Block (Low, 30 min)", f"expC/participant_{participant_id}/expC_experimental_low_30min.txt"))
    else:
        blocks.append(("Experimental Block (High, 30 min)", f"expC/participant_{participant_id}/expC_experimental_high_30min.txt"))
    return blocks


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_calibration_order(rel_scenario_path: str) -> list[float]:
    scenario_path = SCENARIOS_DIR / rel_scenario_path
    text = scenario_path.read_text(encoding="utf-8")

    match = re.search(r"^#\s*m_order\s*=\s*(\[[^\]]+\])\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not parse calibration m_order from scenario header: {rel_scenario_path}")

    try:
        parsed = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError) as exc:
        raise RuntimeError(f"Invalid m_order format in scenario header: {rel_scenario_path}") from exc

    if not isinstance(parsed, list) or len(parsed) == 0:
        raise RuntimeError(f"Calibration m_order must be a non-empty list: {rel_scenario_path}")

    return [float(x) for x in parsed]


def compute_composite_score(perf, prompts, keys, start_t: float = 0.0, end_t: float | None = None) -> dict[str, float]:
    if end_t is None:
        end_t = float("inf")

    perf_w = [r for r in perf if start_t <= r["t"] < end_t]
    prompts_w = [p for p in prompts if start_t <= p["t"] < end_t]
    keys_w = [k for k in keys if start_t <= k["t"] < end_t]

    cursor_vals = [int(r["val"]) for r in perf_w if r["mod"] == "track" and r["addr"] == "cursor_in_target" and r["val"].isdigit()]
    track_acc = (sum(cursor_vals) / len(cursor_vals)) if cursor_vals else 0.0

    a_vals = [int(r["val"]) for r in perf_w if r["mod"] == "resman" and r["addr"] == "a_in_tolerance" and r["val"].isdigit()]
    b_vals = [int(r["val"]) for r in perf_w if r["mod"] == "resman" and r["addr"] == "b_in_tolerance" and r["val"].isdigit()]
    rb = a_vals + b_vals
    resman_acc = (sum(rb) / len(rb)) if rb else 0.0

    sys_labels = [r["val"].lower() for r in perf_w if r["mod"] == "sysmon" and r["addr"] == "signal_detection"]
    hits = sys_labels.count("hit")
    fas = sys_labels.count("fa")
    sysmon_acc = ((hits - fas) / len(sys_labels)) if sys_labels else 0.0

    key_pairs = pair_keys(keys_w)
    chits = cfa = 0
    own_total = len([p for p in prompts_w if p["kind"] == "own"])
    for p in prompts_w:
        window = min(p["t"] + 15, end_t)
        candidates = [kp for kp in key_pairs if not kp["matched"] and p["t"] <= kp["t_press"] <= window]
        if p["kind"] == "own":
            if candidates:
                best = min(candidates, key=lambda x: x["t_press"])
                best["matched"] = True
                chits += 1
        else:
            if candidates:
                best = min(candidates, key=lambda x: x["t_press"])
                best["matched"] = True
                cfa += 1
    comms_acc = ((chits - cfa) / own_total) if own_total else 0.0
    overall = float(mean([track_acc, resman_acc, sysmon_acc, comms_acc]))

    return {
        "track": float(track_acc),
        "resman": float(resman_acc),
        "sysmon": float(sysmon_acc),
        "comms": float(comms_acc),
        "overall": overall,
    }


def compute_calibration_segment_scores(csv_path: Path, m_order: list[float]) -> list[dict[str, float]]:
    perf, prompts, keys = read_csv_events(csv_path)
    segments: list[dict[str, float]] = []

    for idx, m_val in enumerate(m_order):
        start_t = idx * CALIBRATION_SEGMENT_SECONDS
        end_t = (idx + 1) * CALIBRATION_SEGMENT_SECONDS
        scores = compute_composite_score(perf, prompts, keys, start_t=start_t, end_t=end_t)
        segments.append(
            {
                "segment": idx + 1,
                "m": float(m_val),
                "start_t": float(start_t),
                "end_t": float(end_t),
                "track": float(scores["track"]),
                "resman": float(scores["resman"]),
                "sysmon": float(scores["sysmon"]),
                "comms": float(scores["comms"]),
                "overall": float(scores["overall"]),
            }
        )

    return segments


def estimate_capacity_from_calibration(
    segment_scores: list[dict[str, float]],
    target_score: float = CAPACITY_TARGET_SCORE,
    alpha: float = CAPACITY_ALPHA,
) -> tuple[float, float, float, dict[str, float]]:
    if target_score <= 0:
        target_score = CAPACITY_TARGET_SCORE

    x_vals = [float(seg["m"]) for seg in segment_scores]
    y_vals = [float(seg["overall"]) for seg in segment_scores]

    if len(x_vals) < 2:
        raise RuntimeError("Need at least two calibration segments to fit capacity model.")

    x_mean = mean(x_vals)
    y_mean = mean(y_vals)
    var_x = sum((x - x_mean) ** 2 for x in x_vals)
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))

    fit_method = "linear"
    if abs(var_x) < 1e-9:
        slope = 0.0
    else:
        slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean

    if abs(slope) < 1e-9:
        fit_method = "nearest-segment-fallback"
        closest = min(segment_scores, key=lambda seg: abs(float(seg["overall"]) - target_score))
        m_star = float(closest["m"])
    else:
        m_star = (target_score - intercept) / slope

    m_star = clamp(m_star, M_MIN, M_MAX)
    m_low = clamp(alpha * m_star, M_MIN, M_MAX)
    m_high = m_star

    fit_info = {
        "fit_method": fit_method,
        "linear_slope": float(round(slope, 6)),
        "linear_intercept": float(round(intercept, 6)),
        "alpha": float(alpha),
    }
    return float(round(m_star, 4)), float(round(m_low, 4)), float(round(m_high, 4)), fit_info


def write_capacity_fit(
    participant_id: int,
    experiment: str,
    segment_scores: list[dict[str, float]],
    fit_info: dict[str, float],
    m_star: float,
    m_low: float,
    m_high: float,
) -> Path:
    participant_dir = SESSIONS_DIR / f"participant_{participant_id}"
    participant_dir.mkdir(parents=True, exist_ok=True)
    out_path = participant_dir / f"{participant_id}_{experiment}_capacity_fit.json"
    payload = {
        "participant_id": participant_id,
        "experiment": experiment,
        "fit_source": "calibration",
        "target_score": CAPACITY_TARGET_SCORE,
        "calibration_segment_seconds": CALIBRATION_SEGMENT_SECONDS,
        "calibration_segments": segment_scores,
        "fit_info": fit_info,
        "estimated_m_star": m_star,
        "estimated_m_low": m_low,
        "estimated_m_high": m_high,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def generate_personalized_experimental_scenarios(experiment: str, participant_id: int, m_low: float, m_high: float) -> dict[str, str]:
    now = "auto-capacity"

    if experiment == "B":
        from scenario_generators.expB import ExpBGenerator

        gen = ExpBGenerator()
        p_dir = (SCENARIOS_DIR / "expB" / f"participant_{participant_id}")
        low_rel = f"expB/participant_{participant_id}/expB_experimental_low_auto_25min.txt"
        high_rel = f"expB/participant_{participant_id}/expB_experimental_high_auto_25min.txt"

        gen.write_scenario_file(
            p_dir / "expB_experimental_low_auto_25min.txt",
            gen._build_full_block(duration_sec=1500, m=m_low, instruction_file="default/full.txt", include_nasa_tlx=True),
            [f"# Stream B low block participant {participant_id}", f"# auto m_low={m_low}", f"# source={now}"],
        )
        gen.write_scenario_file(
            p_dir / "expB_experimental_high_auto_25min.txt",
            gen._build_full_block(duration_sec=1500, m=m_high, instruction_file="default/full.txt", include_nasa_tlx=True),
            [f"# Stream B high block participant {participant_id}", f"# auto m_high={m_high}", f"# source={now}"],
        )
        return {"low": low_rel, "high": high_rel}

    if experiment == "C":
        from scenario_generators.expC import ExpCGenerator

        gen = ExpCGenerator()
        p_dir = (SCENARIOS_DIR / "expC" / f"participant_{participant_id}")
        low_rel = f"expC/participant_{participant_id}/expC_experimental_low_auto_30min.txt"
        high_rel = f"expC/participant_{participant_id}/expC_experimental_high_auto_30min.txt"

        gen.write_scenario_file(
            p_dir / "expC_experimental_low_auto_30min.txt",
            gen._build_full_block(duration_sec=1800, m=m_low, instruction_file="default/full.txt", include_nasa_tlx=True),
            [f"# Stream C low block participant {participant_id}", f"# auto m_low={m_low}", f"# source={now}"],
        )
        gen.write_scenario_file(
            p_dir / "expC_experimental_high_auto_30min.txt",
            gen._build_full_block(duration_sec=1800, m=m_high, instruction_file="default/full.txt", include_nasa_tlx=True),
            [f"# Stream C high block participant {participant_id}", f"# auto m_high={m_high}", f"# source={now}"],
        )
        return {"low": low_rel, "high": high_rel}

    return {}


def apply_personalized_paths(blocks: list[tuple[str, str]], personalized: dict[str, str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name, rel in blocks:
        new_rel = rel
        if "_experimental_low_" in rel and "low" in personalized:
            new_rel = personalized["low"]
        elif "_experimental_high_" in rel and "high" in personalized:
            new_rel = personalized["high"]
        out.append((name, new_rel))
    return out


def validate_block_files(blocks: list[tuple[str, str]]) -> None:
    missing = [rel for _, rel in blocks if not (SCENARIOS_DIR / rel).exists()]
    if missing:
        msg = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Some scenario files are missing. Run the scenario generator first. Missing files:\n"
            + msg
        )


def get_session_csv_set() -> set[Path]:
    return set(SESSIONS_DIR.glob("**/*.csv"))


def get_face_video_set() -> set[Path]:
    return set(SESSIONS_DIR.glob("**/*_facecamera.mp4"))


def is_primary_session_csv(path: Path) -> bool:
    stem = path.stem
    first = stem.split("_")[0] if "_" in stem else ""
    return first.isdigit()


def relocate_block_log(participant_id: int, rel_scenario_path: str, before_set: set[Path]) -> Path:
    after_set = get_session_csv_set()
    created = sorted(
        [p for p in (after_set - before_set) if is_primary_session_csv(p)],
        key=lambda p: p.stat().st_mtime,
    )

    # Fallback: if no brand-new file is detected (edge case), choose the most recently modified csv
    if not created:
        all_csv = sorted([p for p in after_set if is_primary_session_csv(p)], key=lambda p: p.stat().st_mtime)
        if not all_csv:
            raise FileNotFoundError("No session CSV file was produced for the completed block.")
        source = all_csv[-1]
    else:
        source = created[-1]

    participant_dir = SESSIONS_DIR / f"participant_{participant_id}"
    participant_dir.mkdir(parents=True, exist_ok=True)

    block_stem = Path(rel_scenario_path).stem
    target = participant_dir / f"{participant_id}_{block_stem}.csv"

    # Avoid overwrite if same participant reruns a block
    if target.exists():
        suffix = 2
        while True:
            candidate = participant_dir / f"{participant_id}_{block_stem}_{suffix}.csv"
            if not candidate.exists():
                target = candidate
                break
            suffix += 1

    return Path(shutil.move(str(source), str(target)))


def relocate_block_face_video(participant_id: int, rel_scenario_path: str, before_set: set[Path]) -> Path | None:
    after_set = get_face_video_set()
    created = sorted(list(after_set - before_set), key=lambda p: p.stat().st_mtime)

    if not created:
        return None

    source = created[-1]
    participant_dir = SESSIONS_DIR / f"participant_{participant_id}"
    participant_dir.mkdir(parents=True, exist_ok=True)

    block_stem = Path(rel_scenario_path).stem
    target = participant_dir / f"{participant_id}_{block_stem}_facecamera.mp4"

    if target.exists():
        suffix = 2
        while True:
            candidate = participant_dir / f"{participant_id}_{block_stem}_facecamera_{suffix}.mp4"
            if not candidate.exists():
                target = candidate
                break
            suffix += 1

    return Path(shutil.move(str(source), str(target)))


def run_block(block_name: str, rel_scenario_path: str, idx: int, total: int) -> bool:
    print(f"\n[{idx}/{total}] {block_name}")
    print(f"Scenario: {rel_scenario_path}")
    choice = input("Press Enter to launch, type 's' to skip this block: ").strip().lower()
    if choice == "s":
        print(f"Skipped: {block_name}")
        return False

    # Keep config.ini in sync with what is being launched
    set_openmatb_value("scenario_path", rel_scenario_path)

    # main.py handles one block and saves that block's results when it ends
    subprocess.run([sys.executable, "main.py"], cwd=ROOT, check=True)
    print(f"Completed: {block_name}")
    return True


def read_csv_events(path: Path):
    import csv

    perf, prompts, keys = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", " "])
        rdr = csv.DictReader(f, dialect=dialect)
        if rdr.fieldnames is None:
            return perf, prompts, keys
        rdr.fieldnames = [h.strip() for h in rdr.fieldnames]

        for row in rdr:
            typ = row.get("type", "").strip().lower()
            mod = row.get("module", "").strip().lower()
            addr = row.get("address", "").strip()
            val = row.get("value", "").strip()
            try:
                t = float(row.get("scenario_time", 0) or 0)
            except ValueError:
                t = 0.0

            if typ == "performance":
                perf.append({"mod": mod, "addr": addr, "val": val, "t": t})

            if typ == "event" and mod == "communications" and addr == "radioprompt" and val in ("own", "other"):
                prompts.append({"t": t, "kind": val})

            if (
                typ == "input"
                and ((mod == "keyboard" and addr == "SPACE") or (mod == "joystick" and addr.startswith("JOY_BTN_")))
                and val in ("press", "release")
            ):
                keys.append({"t": t, "action": val})

    return perf, prompts, keys


def pair_keys(keys):
    keys = sorted(keys, key=lambda x: x["t"])
    pairs = []
    i = 0
    while i < len(keys):
        if keys[i]["action"] == "press":
            for j in range(i + 1, len(keys)):
                if keys[j]["action"] == "release":
                    dur = keys[j]["t"] - keys[i]["t"]
                    if dur >= 1.5:
                        pairs.append({"t_press": keys[i]["t"], "dur": dur, "matched": False})
                    i = j
                    break
        i += 1
    return pairs


def compute_instruction_accuracies(csv_path: Path) -> dict[str, float]:
    perf, prompts, keys = read_csv_events(csv_path)
    return compute_composite_score(perf, prompts, keys)


def main() -> None:
    experiment, participant_id, original_scenario_path, stream_b_order, stream_c_condition, record_face = load_openmatb_config()

    if experiment == "A":
        blocks = build_exp_a_block_sequence(participant_id)
    elif experiment == "B":
        blocks = build_exp_b_block_sequence(participant_id, stream_b_order)
    elif experiment == "C":
        blocks = build_exp_c_block_sequence(participant_id, stream_c_condition)
    else:
        raise NotImplementedError(f"Experiment '{experiment}' is not supported. Use A, B, or C.")

    validate_block_files(blocks)

    print("OpenMATB experiment runner")
    print(f"Experiment: {experiment}")
    print(f"Participant ID: {participant_id}")
    if experiment == "B":
        print(f"Stream B order setting: {stream_b_order}")
    if experiment == "C":
        print(f"Stream C condition setting: {stream_c_condition}")
    print(f"Face recording: {'enabled' if record_face else 'disabled'}")
    print(f"Total blocks: {len(blocks)}")

    try:
        idx = 1
        total = len(blocks)
        while idx <= total:
            name, rel_path = blocks[idx - 1]

            # Reminder only (non-blocking) between baseline and experimental phases
            if idx > 1:
                prev_rel_path = blocks[idx - 2][1]
                if prev_rel_path.startswith("expA/participant_") and "/expA_pre_" in prev_rel_path and rel_path.startswith("expA/participant_") and "/expA_main_" in rel_path:
                    print("\nReminder: baseline phase is complete. Please take a 5-minute break before experimental blocks.")
                if experiment == "B" and "expB_experimental_" in rel_path and "expB_experimental_" in prev_rel_path:
                    print("\nReminder: take an 8-10 minute break before the next experimental block.")

            before_set = get_session_csv_set()
            before_video_set = get_face_video_set()
            run_rel_path = build_runtime_facecamera_scenario(rel_path) if record_face else rel_path
            ran = run_block(name, run_rel_path, idx, total)
            if not ran:
                idx += 1
                continue

            saved_log = relocate_block_log(participant_id, rel_path, before_set)
            print(f"Saved block results: {saved_log.relative_to(ROOT)}")

            if record_face:
                saved_video = relocate_block_face_video(participant_id, rel_path, before_video_set)
                if saved_video is not None:
                    print(f"Saved face video: {saved_video.relative_to(ROOT)}")

            if rel_path.endswith("_instructions_combined_L.txt"):
                instr = compute_instruction_accuracies(saved_log)
                print("Instruction combined block accuracies:")
                print(f"  Track : {instr['track']:.3f}")
                print(f"  ResMan: {instr['resman']:.3f}")
                print(f"  SysMon: {instr['sysmon']:.3f}")
                print(f"  Comms : {instr['comms']:.3f}")
                print(f"  Average: {instr['overall']:.3f}")

                repeat_choice = input("Repeat instruction session? [y/N]: ").strip().lower()
                if repeat_choice in {"y", "yes"}:
                    print("Repeating instruction session before continuing...")
                    exp_folder = rel_path.split("/")[0]
                    subtasks_rel = f"{exp_folder}/{exp_folder}_instructions_subtasks.txt"
                    combined_rel = f"{exp_folder}/{exp_folder}_instructions_combined_L.txt"
                    # Insert repeated instruction subtasks + combined immediately after current combined block
                    blocks[idx:idx] = [
                        ("Instruction Subtasks (Repeat)", subtasks_rel),
                        ("Instruction Combined (Low Repeat)", combined_rel),
                    ]
                    total = len(blocks)

            # Streams B/C: fit capacity from 8-min calibration and regenerate personalized experimental scenarios.
            if experiment in {"B", "C"} and rel_path.endswith("_calibration_integrated_8min.txt"):
                m_order = parse_calibration_order(rel_path)
                segment_scores = compute_calibration_segment_scores(saved_log, m_order)
                m_star, m_low, m_high, fit_info = estimate_capacity_from_calibration(segment_scores)
                fit_path = write_capacity_fit(participant_id, experiment, segment_scores, fit_info, m_star, m_low, m_high)
                print(
                    f"Calibration-derived capacity estimate: m*={m_star:.3f}, "
                    f"m_low={m_low:.3f}, m_high={m_high:.3f}"
                )
                print(f"Saved capacity fit: {fit_path.relative_to(ROOT)}")

                personalized = generate_personalized_experimental_scenarios(experiment, participant_id, m_low, m_high)
                if personalized:
                    blocks = apply_personalized_paths(blocks, personalized)
                    validate_block_files(blocks)
                    total = len(blocks)
                    print("Updated upcoming experimental blocks with participant-specific auto-calibrated scenarios.")

            idx += 1
    finally:
        # Restore original value to avoid side effects if desired
        set_openmatb_value("scenario_path", original_scenario_path)

    print("\nExperiment session completed.")


if __name__ == "__main__":
    main()
