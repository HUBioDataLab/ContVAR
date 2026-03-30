"""
Identity-aware Phase-0 GO split: map proteins to sequence groups (e.g. UniRef50),
assign groups to train/val/test, filter triplets so anchor/pos/neg share the same split.
"""
from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

SPLIT_JSON_VERSION = 1


def _normalize_pid(pid: str) -> str:
    return pid.strip().upper()


def load_cluster_map_tsv(path: str) -> Dict[str, str]:
    """
    Load protein_id -> group_id from a TSV file.

    Recognized headers (case-insensitive): protein_id/uniprot/accession and
    group_id/cluster/uniref/cluster_id. Otherwise first two columns as (id, group).
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Cluster map not found: {path}")

    protein_to_group: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return protein_to_group

        lower_map = {name.strip().lower(): name for name in fieldnames}
        pid_key = None
        for cand in ("protein_id", "uniprot", "accession", "id"):
            if cand in lower_map:
                pid_key = lower_map[cand]
                break
        if pid_key is None:
            pid_key = fieldnames[0]

        gid_key = None
        for cand in ("group_id", "cluster", "cluster_id", "uniref", "uniref50"):
            if cand in lower_map:
                gid_key = lower_map[cand]
                break
        if gid_key is None and len(fieldnames) > 1:
            gid_key = fieldnames[1]
        if gid_key is None:
            raise ValueError(
                f"Could not find group column in {path}; expected group_id or second column."
            )

        for row in reader:
            raw_pid = row.get(pid_key)
            raw_gid = row.get(gid_key)
            if raw_pid is None or raw_gid is None:
                continue
            pid = _normalize_pid(str(raw_pid))
            gid = str(raw_gid).strip()
            if pid and gid:
                protein_to_group[pid] = gid

    if not protein_to_group:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                if i == 0 and parts[0].lower() in (
                    "protein_id",
                    "uniprot",
                    "accession",
                    "id",
                ):
                    continue
                pid = _normalize_pid(parts[0])
                gid = parts[1].strip()
                if pid and gid:
                    protein_to_group[pid] = gid

    return protein_to_group


def collect_protein_ids_from_go_tsv(tsv_path: str) -> Set[str]:
    """Union of anchor and candidate IDs from a GO semantic similarity TSV."""
    ids: Set[str] = set()
    if not os.path.exists(tsv_path):
        return ids
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.add(_normalize_pid(row["anchor"]))
            ids.add(_normalize_pid(row["candidate"]))
    return ids


def collect_protein_ids_from_go_tsvs(tsv_paths: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for p in tsv_paths:
        out |= collect_protein_ids_from_go_tsv(p)
    return out


def assign_groups_to_splits(
    group_ids: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    """
    Deterministic shuffle of unique groups, then contiguous slices for train/val/test.
    Sizes: floor(n*train_ratio), floor(n*val_ratio), remainder test.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    groups = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    if n == 0:
        return {}

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    n_test = n - n_train - n_val

    out: Dict[str, str] = {}
    i = 0
    for g in groups[i : i + n_train]:
        out[g] = "train"
    i += n_train
    for g in groups[i : i + n_val]:
        out[g] = "val"
    i += n_val
    for g in groups[i : i + n_test]:
        out[g] = "test"
    return out


def build_protein_to_split(
    protein_to_group: Dict[str, str], group_to_split: Dict[str, str]
) -> Dict[str, str]:
    """Map each protein to train/val/test via its group."""
    out: Dict[str, str] = {}
    for pid, gid in protein_to_group.items():
        sp = group_to_split.get(gid)
        if sp is not None:
            out[pid] = sp
    return out


def split_dict_for_proteins_in_tsvs(
    protein_to_group: Dict[str, str],
    tsv_protein_ids: Set[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Restrict to groups that appear among TSV proteins (that have a cluster mapping),
    then assign splits at group level.
    Returns (group_to_split, sorted list of groups used).
    """
    groups_for_data: Set[str] = set()
    for pid in tsv_protein_ids:
        gid = protein_to_group.get(pid)
        if gid is not None:
            groups_for_data.add(gid)
    sorted_groups = sorted(groups_for_data)
    group_to_split = assign_groups_to_splits(
        sorted_groups, seed, train_ratio, val_ratio, test_ratio
    )
    return group_to_split, sorted_groups


def save_split_json(
    path: str,
    group_to_split: Dict[str, str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cluster_map_path: Optional[str] = None,
    extra_meta: Optional[dict] = None,
) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "version": SPLIT_JSON_VERSION,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "group_to_split": group_to_split,
        "meta": {
            "n_groups": len(group_to_split),
            "cluster_map_path": cluster_map_path,
            **(extra_meta or {}),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ver = data.get("version", 0)
    if ver != SPLIT_JSON_VERSION:
        raise ValueError(f"Unsupported split JSON version {ver}, expected {SPLIT_JSON_VERSION}")
    return data


def filter_triplets_by_split(
    triplets: List[Tuple[str, str, str]],
    protein_to_split: Dict[str, str],
    split_name: str,
) -> List[Tuple[str, str, str]]:
    """Keep triplets where anchor, positive, negative all map to split_name."""
    out: List[Tuple[str, str, str]] = []
    for a, p, n in triplets:
        sa = protein_to_split.get(_normalize_pid(a))
        sp = protein_to_split.get(_normalize_pid(p))
        sn = protein_to_split.get(_normalize_pid(n))
        if sa == sp == sn == split_name:
            out.append((a, p, n))
    return out


def resolve_phase0_split(
    cfg,
    mf_tsv: str,
    bp_tsv: str,
    cc_tsv: str,
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """
    If go_split_mode == 'identity_grouped', return (protein_to_split, group_to_split).
    Otherwise (None, None).
    """
    mode = getattr(cfg, "go_split_mode", "none")
    if mode != "identity_grouped":
        return None, None

    map_path = getattr(cfg, "go_cluster_map_path", None)
    if not map_path:
        raise ValueError(
            "go_split_mode='identity_grouped' requires go_cluster_map_path (protein→group TSV)."
        )

    protein_to_group = load_cluster_map_tsv(map_path)
    tsv_ids = collect_protein_ids_from_go_tsvs([mf_tsv, bp_tsv, cc_tsv])

    json_path = getattr(cfg, "go_split_json_path", None)
    save_path = getattr(cfg, "go_save_split_json_path", None)
    seed = int(getattr(cfg, "go_split_seed", 42))
    rt = float(getattr(cfg, "go_train_ratio", 0.8))
    rv = float(getattr(cfg, "go_val_ratio", 0.1))
    rte = float(getattr(cfg, "go_test_ratio", 0.1))

    if json_path and os.path.exists(json_path):
        data = load_split_json(json_path)
        group_to_split = data["group_to_split"]
        print(f"[Phase0] Loaded group split from {json_path} ({len(group_to_split)} groups).")
    else:
        group_to_split, _ = split_dict_for_proteins_in_tsvs(
            protein_to_group, tsv_ids, seed, rt, rv, rte
        )
        out_path = save_path or json_path
        if out_path:
            save_split_json(
                out_path,
                group_to_split,
                seed,
                rt,
                rv,
                rte,
                cluster_map_path=map_path,
            )
            print(f"[Phase0] Wrote new group split to {out_path}")

    protein_to_split = build_protein_to_split(protein_to_group, group_to_split)
    return protein_to_split, group_to_split
