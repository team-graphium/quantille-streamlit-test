# inference.py
# Centroid-alapú faktor- és profil-kiértékelés (modellfüggő centroid-számítással)

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


LOW = "LOW"
HIGH = "HIGH"


# =========================
#   EMBEDDING HELPER
# =========================

def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
    progress: bool = False,
) -> np.ndarray:
    """
    Wrapper a SentenceTransformer.encode köré.
    """
    show_bar = progress and (tqdm is not None)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_bar,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return emb.astype("float32")


# =========================
#   FAKTOR LEÍRÁSOK ÉS CENTROIDOK
# =========================

def load_factor_passages(
    jsonl_path: str,
    factors: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Betölti a faktor-leírásokat a train/test JSON-ból.

    Várakozás (mint a train kódban):
      raw = json.loads(Path(jsonl_path).read_text())
      minden record:
        - factor_shortage
        - factor_characteristic_low:  [szöveg, ...]
        - factor_characteristic_high: [szöveg, ...]

    Vissza:
      {
        "CSAP_LOW": [...passages...],
        "CSAP_HIGH": [...passages...],
        ...
      }
    """
    raw = json.loads(Path(jsonl_path).read_text(encoding="utf-8"))
    cluster_passages: Dict[str, List[str]] = {}

    for rec in raw:
        f = rec["factor_shortage"]
        if factors is not None and f not in factors:
            continue

        lows = [t.strip() for t in (rec.get("factor_characteristic_low") or []) if t.strip()]
        highs = [t.strip() for t in (rec.get("factor_characteristic_high") or []) if t.strip()]

        if lows:
            cluster_passages[f"{f}_LOW"] = lows
        if highs:
            cluster_passages[f"{f}_HIGH"] = highs

    if not cluster_passages:
        raise ValueError("Nem találtunk egyetlen LOW/HIGH faktor-leírást sem.")

    return cluster_passages


def compute_factor_centroids(
    model: SentenceTransformer,
    jsonl_path: str,
    factors: Optional[List[str]] = None,
    batch_size: int = 32,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    LOW/HIGH centroidok kiszámítása az adott MODELLEL.

    - Minden "F_LOW" / "F_HIGH" labelhez átlagoljuk a hozzá tartozó passage embeddingeket.
    - Embeddingek: "passage: {szöveg}"
    """
    cluster_passages = load_factor_passages(jsonl_path, factors=factors)

    texts: List[str] = []
    labels: List[str] = []

    for label, passages in cluster_passages.items():
        for p in passages:
            texts.append(f"passage: {p}")
            labels.append(label)

    if not texts:
        raise ValueError("Nincs egyetlen szöveg sem a centroid-számításhoz.")

    # Embed az összes szöveg egyszerre (batchelve)
    embs = encode_texts(
        model,
        texts,
        batch_size=batch_size,
        normalize=True,
        progress=progress,
    )  # shape: (N, d)

    # Átlagolás labelenként
    centroids: Dict[str, np.ndarray] = {}
    label_to_sum: Dict[str, np.ndarray] = {}
    label_to_count: Dict[str, int] = {}

    for lab, emb in zip(labels, embs):
        if lab not in label_to_sum:
            label_to_sum[lab] = emb.copy()
            label_to_count[lab] = 1
        else:
            label_to_sum[lab] += emb
            label_to_count[lab] += 1

    for lab, s in label_to_sum.items():
        c = s / max(1, label_to_count[lab])
        # normalizálás
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        centroids[lab] = c.astype("float32")

    return centroids


def build_factor_to_low_high(
    centroids: Dict[str, np.ndarray]
) -> Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    "CSAP_LOW" / "CSAP_HIGH" → faktoronként (low, high) tuple.
    """
    mapping: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

    for label, vec in centroids.items():
        parts = label.split("_", 1)
        if len(parts) != 2:
            continue
        factor, pol = parts
        low_vec, high_vec = mapping.get(factor, (None, None))

        if pol == LOW:
            low_vec = vec
        elif pol == HIGH:
            high_vec = vec
        else:
            continue

        mapping[factor] = (low_vec, high_vec)

    return mapping


# =========================
#   FAKTOR-SCORE SZÁMÍTÁS
# =========================
def estimate_factor_scores_for_text(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    text: str,
    pos_calib: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Egy szövegre faktoronként becsli:
      - rel: max(cos(pass, F_LOW), cos(pass, F_HIGH)) ∈ [-1, 1]
      - pos: becsült 1–8 szint low–high tengelyen (faktoronként kalibráltan, ha pos_calib adott)
      - margin: |s_high - s_low|, polaritás-bizonyosság (0 = bizonytalan low/high irányban)

    Visszatérés:
      {
        "CSAP": {"rel": ..., "pos": ..., "s_low": ..., "s_high": ..., "margin": ...},
        ...
      }
    """
    emb = encode_texts(
        model,
        [f"passage: {text}"],
        batch_size=1,
        normalize=True,
        progress=False,
    )[0]

    factor_to_low_high = build_factor_to_low_high(centroids)

    factor_scores: Dict[str, Dict[str, float]] = {}

    for factor, (c_low, c_high) in factor_to_low_high.items():
        if c_low is None and c_high is None:
            continue

        if c_low is not None:
            s_low = float(emb @ c_low)
        else:
            s_low = float("nan")

        if c_high is not None:
            s_high = float(emb @ c_high)
        else:
            s_high = float("nan")

        # Fallback, ha az egyik centroid hiányzik
        if c_low is None or c_high is None:
            base = c_low if c_low is not None else c_high
            rel = float(emb @ base)
            pos = 4.5  # semleges
            margin = 0.0
        else:
            # relevancia: mennyire "valamilyen" az adott faktor irányában
            rel = max(s_low, s_high)

            # low–high tengelyen vett különbség
            diff = s_high - s_low
            margin = abs(diff)

            if pos_calib is not None and factor in pos_calib:
                # faktor-specifikus lineáris kalibráció diff → pos (1–8 körüli)
                a, b = pos_calib[factor]
                pos = a * diff + b
            else:
                # régi default fallback (-2..2 → 1..8)
                pos_raw = (diff + 2.0) / 4.0
                pos_raw = max(0.0, min(1.0, pos_raw))
                pos = 1.0 + 7.0 * pos_raw

            # clamp 1–8
            pos = float(max(1.0, min(8.0, pos)))

        factor_scores[factor] = {
            "rel": float(rel),
            "pos": float(pos),
            "s_low": float(s_low),
            "s_high": float(s_high),
            "margin": float(margin),
        }

    return factor_scores




# =========================
#   FAKTOR-DIAGNOSZTIKA
# =========================

def load_factor_examples_for_debug(
    jsonl_path: str,
    factors: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Debug célú betöltés:
    minden faktorhoz külön-külön példamondatokat ad vissza LOW/HIGH megjelöléssel.

    Kimenet:
      [
        {"factor": "REF", "polarity": "LOW", "text": "..."},
        {"factor": "REF", "polarity": "HIGH", "text": "..."},
        ...
      ]
    """
    raw = json.loads(Path(jsonl_path).read_text(encoding="utf-8"))
    examples: List[Dict[str, str]] = []

    for rec in raw:
        f = rec["factor_shortage"]
        if factors is not None and f not in factors:
            continue

        lows = rec.get("factor_characteristic_low") or []
        highs = rec.get("factor_characteristic_high") or []

        for t in lows:
            t = t.strip()
            if not t:
                continue
            examples.append({
                "factor": f,
                "polarity": LOW,
                "text": t,
            })

        for t in highs:
            t = t.strip()
            if not t:
                continue
            examples.append({
                "factor": f,
                "polarity": HIGH,
                "text": t,
            })

    if not examples:
        raise ValueError("Nem találtunk egyetlen LOW/HIGH példamondatot sem a debughoz.")

    return examples


def _compute_scores_for_examples(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    examples: List[Dict[str, str]],
    pos_calib: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Dict[str, object]]:
    """
    Segédfüggvény:
      - minden példára lefuttatja a factor-score becslést,
      - eltárolja az adott faktorhoz tartozó rel/pos értéket,
      - valamint a faktor relatív helyét (rank) az összes faktor rel-sorrendjében.
    """
    results = []

    for ex in examples:
        text = ex["text"]
        factor = ex["factor"]

        factor_scores = estimate_factor_scores_for_text(
            model,
            centroids,
            text,
            pos_calib=pos_calib,
        )

        if factor not in factor_scores:
            # ha valamiért nincs centroid ehhez a faktorhoz, átugorjuk
            continue

        fs = factor_scores[factor]
        rel = float(fs["rel"])
        pos = float(fs["pos"])

        # faktor relatív rangsora az összes faktor között relevancia alapján
        ordered = sorted(
            factor_scores.items(),
            key=lambda kv: -kv[1]["rel"],
        )
        rank = None
        for idx, (f_name, _) in enumerate(ordered, start=1):
            if f_name == factor:
                rank = idx
                break

        results.append({
            "factor": factor,
            "polarity": ex["polarity"],
            "text": text,
            "rel": rel,
            "pos": pos,
            "rank": rank,
        })

    return results

def _round_pos_to_int(pos: float) -> int:
    """
    Lebegő POS értéket 1–8 közötti egész pozícióra kerekít.

    - 1.0..1.4 → 1
    - 1.5..2.4 → 2
    - ...
    - 7.5..8.0 → 8
    """
    return int(max(1, min(8, round(pos))))

def debug_single_factor_by_positions(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    jsonl_path: str,
    factor: str,
    examples_per_pos: int = 3,
    pos_calib: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """
    Egy faktor debugja pozíciónként (1–8).

    Minden pozícióra (1..8) megpróbál 2-3 példát mutatni
    (alapértelmezés: examples_per_pos=3), relevancia szerint sorba rendezve.

    Látszik:
      - LOW/HIGH jelölés (polarity),
      - pos (kalibrált 1–8),
      - rel,
      - rank (hányadik legerősebb faktor rel szerint).
    """
    # 1) Példamondatok betöltése csak az adott faktorhoz
    examples = load_factor_examples_for_debug(jsonl_path, factors=[factor])

    if not examples:
        print(f"[{factor}] Nincsenek példamondatok.")
        return

    # 2) Inference lefuttatása kalibrált pos-szal
    scored = _compute_scores_for_examples(
        model,
        centroids,
        examples,
        pos_calib=pos_calib,
    )
    if not scored:
        print(f"[{factor}] Egyik példára sem sikerült faktorscore-t számítani.")
        return

    # 3) Gyors összefoglaló
    lows = [e for e in scored if e["polarity"] == LOW]
    highs = [e for e in scored if e["polarity"] == HIGH]

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    mean_pos_low = _mean([e["pos"] for e in lows])
    mean_pos_high = _mean([e["pos"] for e in highs])

    print(f"=== Faktor debug (pozíciónként): {factor} ===")
    print(f"Összes példa: {len(scored)} (LOW: {len(lows)}, HIGH: {len(highs)})")
    print(f"Átlagos POS (LOW):  {mean_pos_low:.2f}")
    print(f"Átlagos POS (HIGH): {mean_pos_high:.2f}")
    print()

    # 4) Binning 1–8 pozíciókra
    by_pos: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(1, 9)}
    for e in scored:
        p_int = _round_pos_to_int(float(e["pos"]))
        by_pos[p_int].append(e)

    # 5) Pozíciónként példák írasa
    for pos_int in range(1, 9):
        bucket = by_pos.get(pos_int, [])
        if not bucket:
            continue

        # relevancia szerint rendezünk, hogy a “legtipikusabb” jöjjön először
        bucket_sorted = sorted(bucket, key=lambda e: -e["rel"])

        print(f"-- Pozíció {pos_int} (n={len(bucket_sorted)}) --")
        for e in bucket_sorted[:examples_per_pos]:
            snippet = e["text"] if len(e["text"]) <= 120 else e["text"][:117] + "..."
            print(
                f"pol={e['polarity']}, "
                f"pos={e['pos']:.2f}, "
                f"rel={e['rel']:.2f}, "
                f"rank={e['rank']}"
            )
            print(f"  {snippet}")
            print()

def debug_single_factor_by_positions_obj(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    jsonl_path: str,
    factor: str,
    examples_per_pos: int = 3,
    pos_calib: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, object]:
    """
    A korábbi debug_single_factor_by_positions objektumot ad vissza print helyett.
    """

    # 1) Példamondatok csak adott faktorhoz
    examples = load_factor_examples_for_debug(jsonl_path, factors=[factor])
    if not examples:
        return {
            "factor": factor,
            "error": "No examples found",
        }

    # 2) Inference lefuttatása
    scored = _compute_scores_for_examples(
        model,
        centroids,
        examples,
        pos_calib=pos_calib,
    )
    if not scored:
        return {
            "factor": factor,
            "error": "Could not compute scores",
        }

    # 3) Összefoglaló értékek
    lows = [e for e in scored if e["polarity"] == LOW]
    highs = [e for e in scored if e["polarity"] == HIGH]

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    mean_pos_low = _mean([e["pos"] for e in lows])
    mean_pos_high = _mean([e["pos"] for e in highs])

    # 4) Binning 1–8 pozíciókra
    by_pos: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(1, 9)}
    for e in scored:
        p_int = _round_pos_to_int(float(e["pos"]))
        by_pos[p_int].append(e)

    # 5) Objektum építése
    result = {
        "factor": factor,
        "total_examples": len(scored),
        "low_count": len(lows),
        "high_count": len(highs),
        "mean_pos_low": mean_pos_low,
        "mean_pos_high": mean_pos_high,
        "positions": {}
    }

    # 6) Pozíciók feltöltése legfontosabb példákkal
    for pos_int in range(1, 9):
        bucket = by_pos.get(pos_int, [])
        if not bucket:
            continue

        bucket_sorted = sorted(bucket, key=lambda e: -e["rel"])
        selected = bucket_sorted[:examples_per_pos]

        # Csak a lényégi mezők menjenek vissza
        clean_items = []
        for e in selected:
            clean_items.append({
                "polarity": e["polarity"],
                "pos": float(e["pos"]),
                "rel": float(e["rel"]),
                "rank": int(e["rank"]),
                "text": e["text"],
            })

        result["positions"][pos_int] = clean_items

    return result


# =========================
#   POS KALIBRÁCIÓ
# =========================

def compute_factor_pos_calibration(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    jsonl_path: str,
    factors: Optional[List[str]] = None,
    low_target: float = 2.0,
    high_target: float = 7.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Faktoronként kiszámolja az (a, b) lineáris transzformációt a diff → pos leképezéshez.

    Cél:
      LOW példák átlaga ≈ low_target,
      HIGH példák átlaga ≈ high_target.

    Vissza:
      { "CSAP": (a, b), "SZAM": (a, b), ... }
    """
    # Példamondatok betöltése: factor, polarity, text
    raw = json.loads(Path(jsonl_path).read_text(encoding="utf-8"))

    examples: List[Dict[str, str]] = []
    for rec in raw:
        f = rec["factor_shortage"]
        if factors is not None and f not in factors:
            continue

        lows = rec.get("factor_characteristic_low") or []
        highs = rec.get("factor_characteristic_high") or []

        for t in lows:
            txt = t.strip()
            if not txt:
                continue
            examples.append({"factor": f, "polarity": LOW, "text": txt})

        for t in highs:
            txt = t.strip()
            if not txt:
                continue
            examples.append({"factor": f, "polarity": HIGH, "text": txt})

    if not examples:
        raise ValueError("Nem találtunk példamondatokat a kalibrációhoz.")

    # Faktoronként gyűjtjük a diff értékeket (s_high - s_low)
    diff_by_factor_low: Dict[str, List[float]] = {}
    diff_by_factor_high: Dict[str, List[float]] = {}

    for ex in examples:
        text = ex["text"]
        factor = ex["factor"]

        # egyetlen szöveg factor-score-ja (csak diff kell)
        scores = estimate_factor_scores_for_text(
            model,
            centroids,
            text,
            pos_calib=None,  # itt NYERS diff kell
        )
        fs = scores.get(factor)
        if fs is None:
            continue

        s_low = float(fs["s_low"])
        s_high = float(fs["s_high"])

        if np.isnan(s_low) or np.isnan(s_high):
            continue

        diff = s_high - s_low

        if ex["polarity"] == LOW:
            diff_by_factor_low.setdefault(factor, []).append(diff)
        else:
            diff_by_factor_high.setdefault(factor, []).append(diff)

    # Faktoronként lineáris transzformáció (a, b)
    calib: Dict[str, Tuple[float, float]] = {}

    for factor in sorted(set(list(diff_by_factor_low.keys()) + list(diff_by_factor_high.keys()))):
        lows = diff_by_factor_low.get(factor, [])
        highs = diff_by_factor_high.get(factor, [])

        if not lows or not highs:
            # ha nincs mindkét irány, marad az alap skála
            continue

        mean_low = float(np.mean(lows))
        mean_high = float(np.mean(highs))

        # Ha a két átlag túl közel van egymáshoz, ne osszunk nullával
        if abs(mean_high - mean_low) < 1e-4:
            continue

        # Oldjuk meg:
        #   a * mean_low  + b = low_target
        #   a * mean_high + b = high_target
        a = (high_target - low_target) / (mean_high - mean_low)
        b = low_target - a * mean_low

        calib[factor] = (a, b)

    return calib


# ========== SIMPLE LOGIC

from typing import Any

def _profile_alignment_key(
    profile_levels: Dict[str, float],
    fs_all: Dict[str, Dict[str, float]],
    rel_threshold: float = 0.3,
) -> Tuple[int, float, float]:
    """
    Egyszerű profil-illeszkedési kulcs egy mondatra.

    Vissza:
      (n_match, mean_diff, mean_rel)

    - n_match: hány faktor illeszkedik (rel >= rel_threshold),
    - mean_diff: átlagos |pos - level| a match-eknél,
    - mean_rel: átlagos rel a match-eknél.
    """
    diffs: List[float] = []
    rels: List[float] = []

    for factor, level in profile_levels.items():
        fs = fs_all.get(factor)
        if fs is None:
            continue

        rel = float(fs.get("rel", 0.0))
        if rel < rel_threshold:
            continue

        pos = float(fs.get("pos", 4.5))
        diff = abs(pos - level)

        diffs.append(diff)
        rels.append(rel)

    if not diffs:
        # nincs egyetlen érdemi match sem → tegyük a sor végére
        return (0, float("inf"), 0.0)

    n_match = len(diffs)
    mean_diff = float(np.mean(diffs))
    mean_rel = float(np.mean(rels))

    return (n_match, mean_diff, mean_rel)


def compute_factor_scores_for_texts(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    texts: List[str],
    pos_calib: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Dict[str, Dict[str, float]]]:
    """
    Minden szövegre lefuttatja az estimate_factor_scores_for_text-et.
    Kimenet: listában, szöveg-sorrendben.

    [
      {"CSAP": {...}, "KAP": {...}, ...},   # 0. szöveg
      {"CSAP": {...}, "KAP": {...}, ...},   # 1. szöveg
      ...
    ]
    """
    all_scores: List[Dict[str, Dict[str, float]]] = []
    for t in texts:
        fs = estimate_factor_scores_for_text(
            model,
            centroids,
            t,
            pos_calib=pos_calib,
        )
        all_scores.append(fs)
    return all_scores


def _pos_to_bin(pos: float) -> int:
    """
    POS 1.0..8.0 → 1..8 egész bin.

    Itt nem csúsztatunk, simán kerekítünk:
      1.0–1.4 → 1, 1.5–2.4 → 2, ..., 7.5–8.0 → 8
    """
    return int(max(1, min(8, round(pos))))


def prebin_texts_by_factor(
    texts: List[str],
    all_scores: List[Dict[str, Dict[str, float]]],
    rel_threshold: float = 0.3,
    top_k_factors: int = 3,
) -> Dict[str, Dict[int, List[Tuple[str, Dict[str, float], Dict[str, Dict[str, float]]]]]]:
    """
    Faktoronként és POS-binenként elrendezi a szövegeket:

      by_factor[f][bin] = [(text, fs_factor, fs_all), ...]

    Csak azok a (text, factor) párok kerülnek be, ahol:
      - van factor-score,
      - rel >= rel_threshold,
      - a faktor benne van a szöveg top_k_factors releváns faktorai között.
    """

    by_factor: Dict[str, Dict[int, List[Tuple[str, Dict[str, float], Dict[str, Dict[str, float]]]]]] = {}

    for text, fs_all in zip(texts, all_scores):
        # top-k faktorokat megnézzük relevancia alapján
        ordered = sorted(
            fs_all.items(),
            key=lambda kv: -float(kv[1].get("rel", 0.0)),
        )
        top_factors = [f_name for f_name, _ in ordered[:top_k_factors]]

        for factor, fs in fs_all.items():
            if factor not in top_factors:
                continue

            rel = float(fs.get("rel", 0.0))
            if rel < rel_threshold:
                continue

            pos = float(fs.get("pos", 4.5))
            bin_id = _pos_to_bin(pos)

            factor_bins = by_factor.setdefault(factor, {})
            bucket = factor_bins.setdefault(bin_id, [])
            bucket.append((text, fs, fs_all))

    # alap sorting: faktor saját relevanciája szerint
    for factor, bins in by_factor.items():
        for bin_id, bucket in bins.items():
            bins[bin_id] = sorted(
                bucket,
                key=lambda pair: -float(pair[1].get("rel", 0.0)),  # pair[1] = fs_factor
            )

    return by_factor


def _bins_around_target(
    target_bin: int,
    max_bins: int = 8,
) -> List[int]:
    """
    Egyszerű helper: egy cél bin körül (target_bin) generál egy prioritási sorrendet,
    pl. target=6 → [6, 5, 7, 4, 8, 3, 2, 1]

    Ezt akkor használjuk, ha az adott binben kevés jelölt van,
    és kicsit ki akarunk tágítani ugyanazon oldal környékére.
    """
    bins = list(range(1, max_bins + 1))
    # sorbarakjuk távolság szerint
    bins.sort(key=lambda b: abs(b - target_bin))
    return bins

def _transform_alignment_key(alignment: Tuple[int, float, float]) -> Tuple[int, float, float]:
    """
    alignment = (n_match, mean_diff, mean_rel)

    Rendezéshez szeretnénk:
      - n_match: csökkenő (nagyobb jobb)  → -n_match
      - mean_diff: növekvő (kisebb jobb)  → mean_diff
      - mean_rel: csökkenő (nagyobb jobb) → -mean_rel
    """
    n_match, mean_diff, mean_rel = alignment
    return (-n_match, mean_diff, -mean_rel)

def sample_texts_for_profile_simple(
    profile_levels: Dict[str, float],
    texts: List[str],
    all_scores: List[Dict[str, Dict[str, float]]],
    rel_threshold: float = 0.3,
    top_k_factors: int = 3,
    n_extreme: int = 3,
    n_mid: int = 1,
    rerank_by_profile: bool = True,
    rerank_rel_threshold: float = 0.3,
    min_rel_for_rerank: Optional[float] = 0.6,
    top_n_per_bin_before_rerank: Optional[int] = 5,
) -> Dict[str, List[Tuple[str, int, Dict[str, float], str]]]:
    """
    Egyszerű, faktoronkénti mintavétel egy profil alapján, kiegészítve:

    - Bin: a faktor POS-hoz illeszkedő tartomány.
    - Bin jelöltjei:
        1) alapból rel szerint válogatva,
        2) opcionálisan:
            - min_rel_for_rerank alapján beszűkítjük,
            - majd top_n_per_bin_before_rerank jelöltet megtartunk,
            - ezeket profil-alignment szerint újrarendezzük.

    Vissza:
      {
        "CSAP": [
          (text, bin, fs_factor, "LOW"/"MID"/"HIGH"),
          ...
        ],
        ...
      }
    """
    by_factor = prebin_texts_by_factor(
        texts=texts,
        all_scores=all_scores,
        rel_threshold=rel_threshold,
        top_k_factors=top_k_factors,
    )

    result: Dict[str, List[Tuple[str, int, Dict[str, float], str]]] = {}

    for factor, level in profile_levels.items():
        factor_bins = by_factor.get(factor, {})
        if not factor_bins:
            continue

        selected: List[Tuple[str, int, Dict[str, float], str]] = []

        if level <= 3.0:
            target_bin = _pos_to_bin(level)
            bins_order = _bins_around_target(target_bin)
            bins_order = [b for b in bins_order if b <= 4] + [b for b in bins_order if b > 4]
            needed = n_extreme
            label = "LOW"

        elif level >= 6.0:
            target_bin = _pos_to_bin(level)
            bins_order = _bins_around_target(target_bin)
            bins_order = [b for b in bins_order if b >= 5] + [b for b in bins_order if b < 5]
            needed = n_extreme
            label = "HIGH"

        else:
            target_bin = _pos_to_bin(level)
            bins_order = _bins_around_target(target_bin)
            bins_order = [b for b in bins_order if 3 <= b <= 6]
            needed = n_mid
            label = "MID"

        picked = 0
        seen_texts = set()

        for b in bins_order:
            if picked >= needed:
                break
            bucket = factor_bins.get(b, [])
            if not bucket:
                continue

            # bucket: [(text, fs_factor, fs_all), ...]
            candidates = bucket

            # 1) Szűkítés min_rel_for_rerank alapján (ha be van állítva)
            if min_rel_for_rerank is not None:
                filtered = [
                    (text, fs_factor, fs_all)
                    for (text, fs_factor, fs_all) in candidates
                    if float(fs_factor.get("rel", 0.0)) >= min_rel_for_rerank
                ]
                if filtered:
                    candidates = filtered  # csak akkor cseréljük, ha nem ürül ki

            # 2) Top-N faktor-relevancia szerint (ha be van állítva)
            if top_n_per_bin_before_rerank is not None and len(candidates) > top_n_per_bin_before_rerank:
                candidates = sorted(
                    candidates,
                    key=lambda pair: -float(pair[1].get("rel", 0.0)),  # fs_factor rel
                )[:top_n_per_bin_before_rerank]

            # 3) Profil-alapú rerank (ha engedélyezett)
            if rerank_by_profile:
                candidates = sorted(
                    candidates,
                    key=lambda pair: _transform_alignment_key(
                        _profile_alignment_key(
                            profile_levels=profile_levels,
                            fs_all=pair[2],  # fs_all
                            rel_threshold=rerank_rel_threshold,
                        )
                    ),
                )

            # 4) Jelöltek kiválasztása ebből a binből
            for text, fs_factor, fs_all in candidates:
                if picked >= needed:
                    break
                if text in seen_texts:
                    continue
                selected.append((text, b, fs_factor, label))
                seen_texts.add(text)
                picked += 1

        result[factor] = selected

    return result
