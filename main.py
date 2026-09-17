"""
PIPELINE
--------
1. User enters a batter name, a pitcher name, and N.
2. Resolve both names to MLB IDs, get the batter's career date range, and
   fetch the batter's ENTIRE career of pitch-by-pitch data ONCE (cached) --
   reused for both the faced-pitcher list and every comp's stat line, so
   we never re-download the same career data twice.
3. Build the arsenal fingerprint matrix -- one row per pitcher (the target
   + every active pitcher the batter has faced), same features for all.
4. Weighted Euclidean distance from the target pitcher to every other
   pitcher in the matrix; take the N closest.
5. Pull the batter's real historical PA/AVG/SLG/K% against each of those
   N comps (from the already-fetched career data -- no network calls),
   and blend them into one reliability-weighted performance summary.
"""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import statsapi

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

STANDARD_PITCH_TYPES = [
    "FF",  # 4-seam fastball
    "SI",  # sinker
    "FC",  # cutter
    "SL",  # slider
    "ST",  # sweeper
    "CU",  # curveball
    "KC",  # knuckle curve
    "CH",  # changeup
    "FS",  # splitter
    "OTHER",
]
FEATURES = ["usage_pct", "avg_velo", "avg_spin", "avg_h_break", "avg_v_break"]

MIN_USAGE_TO_COUNT = 0.03        # pitches thrown less than this often get folded into OTHER
MIN_PITCHES_FOR_ARSENAL = 100    # below this, we don't trust the arsenal numbers
PA_CAP_FOR_FULL_WEIGHT = 20      # PA above this get no extra weight in the blend

ARSENAL_LOOKBACK_DAYS = 730      # arsenal fingerprints use ~2 recent seasons, not a career-long
                                  # average -- reflects a pitcher's CURRENT stuff, and is a much
                                  # smaller (faster) download than their whole career
MAX_CANDIDATES = 150             # cap on how many comp pitchers we ever fetch arsenals for;
                                  # ranked by most recent matchup, so the cap keeps the most
                                  # relevant/current comps if a career has more than this
PARALLEL_FETCH_WORKERS = 8       # arsenal fetches are network-bound waits, not CPU work, so
                                  # running several concurrently cuts wall-clock time a lot

CACHE_DIR = "arsenal_cache"

def get_player_id(name: str) -> int:
    matches = statsapi.lookup_player(name)
    if not matches:
        raise ValueError(f"No player found matching '{name}'")
    if len(matches) > 1:
        print(f"  (multiple matches for '{name}', using: {matches[0]['fullName']})")
    return matches[0]["id"]


def get_batter_career_range(batter_id: int) -> tuple[str, str]:
    data = statsapi.get("people", {"personIds": batter_id})
    person = data["people"][0]
    start_dt = person["mlbDebutDate"]
    end_dt = date.today().isoformat()
    return start_dt, end_dt


def get_all_faced_pitchers(pitches: pd.DataFrame) -> list[int]:
    return [int(pid) for pid in pitches["pitcher"].unique()]


def get_arsenal_window(end_dt: str) -> tuple[str, str]:
    end = date.fromisoformat(end_dt)
    start = end - timedelta(days=ARSENAL_LOOKBACK_DAYS)
    return start.isoformat(), end_dt


def rank_by_recency_and_cap(pitches: pd.DataFrame, pitcher_ids: list[int],
                             max_candidates: int = MAX_CANDIDATES) -> list[int]:
    subset = pitches[pitches["pitcher"].isin(pitcher_ids)]
    most_recent = subset.groupby("pitcher")["game_date"].max().sort_values(ascending=False)
    return [int(pid) for pid in most_recent.index[:max_candidates]]


def fetch_arsenals_parallel(pitcher_ids: list[int], arsenal_start_dt: str, arsenal_end_dt: str,
                             fallback_start_dt: str, max_workers: int = PARALLEL_FETCH_WORKERS) -> dict[int, pd.Series]:
    vectors: dict[int, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(get_arsenal_with_fallback, pid, arsenal_start_dt, arsenal_end_dt, fallback_start_dt): pid
            for pid in pitcher_ids
        }
        done = 0
        for future in as_completed(future_to_id):
            pid = future_to_id[future]
            done += 1
            try:
                vectors[pid] = future.result()
            except ValueError as e:
                print(f"  [{done}/{len(pitcher_ids)}] skipping {pid}: {e}")
            else:
                print(f"  [{done}/{len(pitcher_ids)}] fetched arsenal for pitcher {pid}")
    return vectors


def filter_to_active_pitchers(pitcher_ids: list[int], batch_size: int = 100) -> list[int]:
    active_ids = []
    for i in range(0, len(pitcher_ids), batch_size):
        chunk = pitcher_ids[i:i + batch_size]
        id_string = ",".join(str(pid) for pid in chunk)
        data = statsapi.get("people", {"personIds": id_string})
        for person in data["people"]:
            if person.get("active"):
                active_ids.append(person["id"])
    return active_ids


def get_pitcher_names(pitcher_ids: list[int], batch_size: int = 100) -> dict[int, str]:
    names = {}
    for i in range(0, len(pitcher_ids), batch_size):
        chunk = pitcher_ids[i:i + batch_size]
        id_string = ",".join(str(pid) for pid in chunk)
        data = statsapi.get("people", {"personIds": id_string})
        for person in data["people"]:
            names[person["id"]] = person["fullName"]
    return names

def raw_pitches_to_arsenal_table(pitches: pd.DataFrame) -> pd.DataFrame:
    df = pitches.copy()
    df["pitch_type"] = df["pitch_type"].fillna("OTHER")
    df.loc[~df["pitch_type"].isin(STANDARD_PITCH_TYPES), "pitch_type"] = "OTHER"

    total_pitches = len(df)
    if total_pitches < MIN_PITCHES_FOR_ARSENAL:
        raise ValueError(f"Only {total_pitches} pitches in sample (need >= {MIN_PITCHES_FOR_ARSENAL})")

    def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby("pitch_type").agg(
            n_pitches=("pitch_type", "size"),
            avg_velo=("release_speed", "mean"),
            avg_spin=("release_spin_rate", "mean"),
            avg_h_break=("pfx_x", lambda s: s.mean() * 12),
            avg_v_break=("pfx_z", lambda s: s.mean() * 12),
        )
        g["usage_pct"] = round(g["n_pitches"] / total_pitches, 5)
        return g

    grouped = _aggregate(df)

    # Fold rare "show-me" pitches into OTHER, then re-aggregate.
    rare_types = grouped[grouped["usage_pct"] < MIN_USAGE_TO_COUNT].index
    if len(rare_types) and len(rare_types) < len(grouped):
        df.loc[df["pitch_type"].isin(rare_types), "pitch_type"] = "OTHER"
        grouped = _aggregate(df)

    return grouped[["usage_pct", "avg_velo", "avg_spin", "avg_h_break", "avg_v_break"]]


def arsenal_table_to_vector(arsenal: pd.DataFrame) -> pd.Series:
    slots = {}
    for pitch_type in STANDARD_PITCH_TYPES:
        for feature in FEATURES:
            key = f"{pitch_type}_{feature}"
            slots[key] = arsenal.loc[pitch_type, feature] if pitch_type in arsenal.index else np.nan
    return pd.Series(slots)


def build_fingerprint(pitches: pd.DataFrame) -> pd.Series:
    return arsenal_table_to_vector(raw_pitches_to_arsenal_table(pitches))


def fetch_pitcher_arsenal_vector(pitcher_id: int, start_dt: str, end_dt: str) -> pd.Series:
    from pybaseball import statcast_pitcher

    pitches = statcast_pitcher(start_dt, end_dt, pitcher_id)
    return build_fingerprint(pitches)


def get_cached_or_fetch_batter_pitches(batter_id: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"batter_{batter_id}_{start_dt}_{end_dt}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from pybaseball import statcast_batter
    pitches = statcast_batter(start_dt, end_dt, batter_id)

    with open(cache_path, "wb") as f:
        pickle.dump(pitches, f)

    return pitches


def get_cached_or_fetch_arsenal(pitcher_id: int, start_dt: str, end_dt: str) -> pd.Series:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{pitcher_id}_{start_dt}_{end_dt}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    vector = fetch_pitcher_arsenal_vector(pitcher_id, start_dt, end_dt)

    with open(cache_path, "wb") as f:
        pickle.dump(vector, f)

    return vector


def get_arsenal_with_fallback(pitcher_id: int, arsenal_start_dt: str, arsenal_end_dt: str,
                               fallback_start_dt: str) -> pd.Series:
    try:
        return get_cached_or_fetch_arsenal(pitcher_id, arsenal_start_dt, arsenal_end_dt)
    except ValueError:
        return get_cached_or_fetch_arsenal(pitcher_id, fallback_start_dt, arsenal_end_dt)


def compute_pool_stats(fingerprints: pd.DataFrame) -> pd.DataFrame:
    means = fingerprints.mean(skipna=True)
    stds = fingerprints.std(skipna=True).replace(0, np.nan)
    return pd.DataFrame({"mean": means, "std": stds})


def normalize_fingerprint(fingerprint: pd.Series, pool_stats: pd.DataFrame) -> pd.Series:
    return (fingerprint - pool_stats["mean"]) / pool_stats["std"]


def pitcher_distance(vec_a: pd.Series, vec_b: pd.Series, pool_stats: pd.DataFrame) -> float:
    za = normalize_fingerprint(vec_a, pool_stats)
    zb = normalize_fingerprint(vec_b, pool_stats)

    total_sq_dist = 0.0
    total_weight = 0.0

    for pitch_type in STANDARD_PITCH_TYPES:
        usage_key = f"{pitch_type}_usage_pct"
        usage_a = vec_a.get(usage_key, np.nan)
        usage_b = vec_b.get(usage_key, np.nan)
        usage_a = 0.0 if pd.isna(usage_a) else usage_a
        usage_b = 0.0 if pd.isna(usage_b) else usage_b

        pitch_weight = (usage_a + usage_b) / 2.0
        if pitch_weight <= 0:
            continue

        for feature in FEATURES:
            key = f"{pitch_type}_{feature}"

            if feature == "usage_pct":
                mean = pool_stats.loc[usage_key, "mean"]
                std = pool_stats.loc[usage_key, "std"]
                if pd.isna(std):
                    continue
                va = (usage_a - mean) / std
                vb = (usage_b - mean) / std
            else:
                va, vb = za.get(key, np.nan), zb.get(key, np.nan)
                if pd.isna(va) or pd.isna(vb):
                    continue

            total_sq_dist += pitch_weight * (va - vb) ** 2
            total_weight += pitch_weight

    if total_weight == 0:
        return np.inf
    return float(np.sqrt(total_sq_dist / total_weight))


def distance_to_similarity_pct(distance: float, scale: float = 1.5) -> float:
    if np.isinf(distance):
        return 0.0
    return float(100 * np.exp(-distance / scale))


def rank_similar_pitchers(target_id: int, target_vector: pd.Series,
                           candidate_vectors: dict[int, pd.Series], top_n: int = 10) -> pd.DataFrame:
    all_vectors = pd.DataFrame(candidate_vectors).T
    pool_stats = compute_pool_stats(all_vectors)

    rows = []
    for pid, vec in candidate_vectors.items():
        if pid == target_id:
            continue
        d = pitcher_distance(target_vector, vec, pool_stats)
        rows.append({"pitcher_id": pid, "distance": d, "similarity_pct": distance_to_similarity_pct(d)})

    result = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    return result.head(top_n)

def batter_vs_pitcher_line_from_pitches(pitches: pd.DataFrame, pitcher_id: int) -> dict:
    matchup = pitches[pitches["pitcher"] == pitcher_id]
    pa_rows = matchup[matchup["events"].notna()]
    pa = len(pa_rows)
    if pa == 0:
        return {"PA": 0, "AVG": 0.0, "SLG": 0.0, "K_pct": 0.0}

    non_ab_events = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"}
    ab_rows = pa_rows[~pa_rows["events"].isin(non_ab_events)]
    ab = len(ab_rows)

    hit_events = {"single", "double", "triple", "home_run"}
    bases_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

    hits = pa_rows["events"].isin(hit_events).sum()
    total_bases = pa_rows["events"].map(bases_map).fillna(0).sum()
    strikeouts = (pa_rows["events"] == "strikeout").sum()

    avg = hits / ab if ab > 0 else 0.0
    slg = total_bases / ab if ab > 0 else 0.0
    k_pct = 100 * strikeouts / pa

    return {"PA": pa, "AVG": round(avg, 3), "SLG": round(slg, 3), "K_pct": round(k_pct, 1)}


def weighted_batter_profile(comps_with_stats: pd.DataFrame) -> dict:
    df = comps_with_stats.copy()
    df = df[df["PA"] > 0]

    if df.empty:
        return {"AVG": None, "SLG": None, "K_pct": None, "effective_sample": 0.0, "n_comps_used": 0}

    df["similarity_weight"] = df["similarity_pct"] / 100.0
    df["sample_weight"] = df["PA"].clip(upper=PA_CAP_FOR_FULL_WEIGHT) / PA_CAP_FOR_FULL_WEIGHT
    df["weight"] = df["similarity_weight"] * df["sample_weight"]

    total_weight = df["weight"].sum()
    if total_weight == 0:
        return {"AVG": None, "SLG": None, "K_pct": None, "effective_sample": 0.0, "n_comps_used": 0}

    def wavg(col):
        return float((df[col] * df["weight"]).sum() / total_weight)

    return {
        "AVG": round(wavg("AVG"), 3),
        "SLG": round(wavg("SLG"), 3),
        "K_pct": round(wavg("K_pct"), 1),
        "effective_sample": round(total_weight * PA_CAP_FOR_FULL_WEIGHT, 1),
        "n_comps_used": int(len(df)),
    }

# Approximate modern-day MLB benchmarks, used ONLY to translate a number into a plain-language label
LEAGUE_AVG_AVG = 0.245
LEAGUE_AVG_SLG = 0.400
LEAGUE_AVG_K_PCT = 22.5

LOW_SIMILARITY_WARNING_THRESHOLD = 40.0

CONFIDENCE_BANDS = [
    (5.0, "Very low"),
    (15.0, "Low"),
    (30.0, "Moderate"),
    (float("inf"), "Solid"),
]


def confidence_label(effective_sample: float) -> str:
    for cutoff, label in CONFIDENCE_BANDS:
        if effective_sample < cutoff:
            return label
    return CONFIDENCE_BANDS[-1][1]


def relative_label(value: float, benchmark: float) -> str:
    if benchmark == 0:
        return "N/A"
    pct_diff = (value - benchmark) / benchmark

    if pct_diff >= 0.25:
        return "well above average"
    elif pct_diff >= 0.10:
        return "above average"
    elif pct_diff > -0.10:
        return "about average"
    elif pct_diff > -0.25:
        return "below average"
    else:
        return "well below average"


def k_pct_relative_label(k_pct: float) -> str:
    raw_label = relative_label(k_pct, LEAGUE_AVG_K_PCT)
    if raw_label in ("well above average", "above average"):
        return f"{raw_label} (tougher matchup for the batter -- more swing-and-miss risk)"
    elif raw_label in ("well below average", "below average"):
        return f"{raw_label} (easier matchup for the batter -- more contact than usual)"
    return raw_label


def plain_language_summary(profile: dict) -> list[str]:
    if profile["AVG"] is None:
        return ["Not enough data to describe expected performance."]

    avg, slg, k_pct = profile["AVG"], profile["SLG"], profile["K_pct"]
    iso = round(slg - avg, 3)  # isolated power -- extra bases per at-bat, beyond just AVG

    lines = []

    if avg > 0:
        hits_per_10 = round(avg * 10, 1)
        lines.append(
            f"Contact quality: about {hits_per_10} hits per 10 at-bats "
            f"({relative_label(avg, LEAGUE_AVG_AVG)} -- MLB average is roughly {LEAGUE_AVG_AVG:.3f})."
        )
    else:
        lines.append("Contact quality: no hits in the data used for this projection.")

    lines.append(
        f"Power: {relative_label(slg, LEAGUE_AVG_SLG)} slugging "
        f"(SLG {slg:.3f} vs. a league-average benchmark around {LEAGUE_AVG_SLG:.3f}); "
        f"isolated power (extra bases per at-bat, beyond AVG) works out to about {iso:.3f}."
    )

    if k_pct > 0:
        one_in_n = round(100 / k_pct, 1)
        lines.append(
            f"Strikeout risk: about 1 strikeout every {one_in_n} plate appearances "
            f"({k_pct_relative_label(k_pct)} -- MLB average is roughly {LEAGUE_AVG_K_PCT:.1f}%)."
        )
    else:
        lines.append("Strikeout risk: no strikeouts in the data used for this projection.")

    return lines

\
# OUTPUT

def print_scouting_summary(target_name: str, batter_name: str,
                            comps: pd.DataFrame, profile: dict) -> None:
    print(f"\nScouting report: {batter_name} vs. {target_name} (never faced)")
    print("Based on performance against these similar pitchers:\n")

    for _, row in comps.iterrows():
        print(
            f"  {row['name']:<25} similarity: {row['similarity_pct']:5.1f}%   "
            f"PA: {int(row['PA']):3d}   AVG: {row['AVG']:.3f}   SLG: {row['SLG']:.3f}   K%: {row['K_pct']:.1f}"
        )

    # --- Safeguard: flag it if even the BEST comp is a weak stylistic match ---
    if not comps.empty:
        best_similarity = comps["similarity_pct"].max()
        if best_similarity <= LOW_SIMILARITY_WARNING_THRESHOLD:
            print(
                f"\n\u26a0 Note: the closest comp found is only {best_similarity:.1f}% similar to "
                f"{target_name}. That suggests {target_name}'s style may be fairly unique among "
                f"the pitchers {batter_name} has actually faced -- treat the projection below "
                f"with reduced confidence, since none of these comps are a strong stylistic match."
            )

    print(f"\nProjected performance for {batter_name} vs. {target_name}'s style:")
    if profile["AVG"] is None:
        print("  Not enough data to make a reliable projection.")
        return

    print(f"  AVG: {profile['AVG']:.3f}   SLG: {profile['SLG']:.3f}   K%: {profile['K_pct']:.1f}")
    conf = confidence_label(profile["effective_sample"])
    print(f"  Confidence: {conf}  (effective sample ~{profile['effective_sample']:.1f} PA "
          f"across {profile['n_comps_used']} comparable pitcher(s))")

    print("\nWhat this means:")
    for line in plain_language_summary(profile):
        print(f"  - {line}")


# LIVE MODE -- scan today's live games, track one, and re-run the scouting

def list_live_games() -> list[dict]:
    today = date.today().isoformat()
    games = statsapi.schedule(date=today)
    return [g for g in games if g.get("status") == "In Progress"]


def choose_live_game() -> int:
    games = list_live_games()
    if not games:
        raise RuntimeError("No games currently in progress.")

    print("\nLive games right now:")
    for i, g in enumerate(games, start=1):
        print(f"  {i}. {g['away_name']} @ {g['home_name']}")

    choice = int(input("Select a game number: ").strip())
    return games[choice - 1]["game_id"]


def get_current_matchup(game_pk: int) -> dict | None:
    data = statsapi.get("game", {"gamePk": game_pk})
    try:
        matchup = data["liveData"]["plays"]["currentPlay"]["matchup"]
    except KeyError:
        return None

    batter = matchup.get("batter")
    pitcher = matchup.get("pitcher")
    if not batter or not pitcher:
        return None

    return {
        "batter_id": batter["id"],
        "batter_name": batter["fullName"],
        "pitcher_id": pitcher["id"],
        "pitcher_name": pitcher["fullName"],
    }


def run_scouting_for_matchup(batter_id: int, batter_name: str, pitcher_id: int,
                              pitcher_name: str, top_n: int, session_cache: dict) -> None:
    if batter_id not in session_cache:
        print(f"  (first time seeing {batter_name} this session -- building their comp pool...)")
        start_dt, end_dt = get_batter_career_range(batter_id)
        arsenal_start_dt, arsenal_end_dt = get_arsenal_window(end_dt)
        batter_pitches = get_cached_or_fetch_batter_pitches(batter_id, start_dt, end_dt)
        faced = get_all_faced_pitchers(batter_pitches)
        active_faced = filter_to_active_pitchers(faced)
        active_faced = rank_by_recency_and_cap(batter_pitches, active_faced)

        candidate_vectors = fetch_arsenals_parallel(active_faced, arsenal_start_dt, arsenal_end_dt, start_dt)

        session_cache[batter_id] = {
            "start_dt": start_dt,
            "end_dt": end_dt,
            "arsenal_start_dt": arsenal_start_dt,
            "arsenal_end_dt": arsenal_end_dt,
            "batter_pitches": batter_pitches,
            "candidate_vectors": candidate_vectors,
        }

    cached = session_cache[batter_id]
    arsenal_start_dt, arsenal_end_dt = cached["arsenal_start_dt"], cached["arsenal_end_dt"]
    candidate_vectors = cached["candidate_vectors"]  # mutated in place -- shared across at-bats for this batter

    if pitcher_id not in candidate_vectors:
        try:
            candidate_vectors[pitcher_id] = get_arsenal_with_fallback(
                pitcher_id, arsenal_start_dt, arsenal_end_dt, cached["start_dt"]
            )
        except ValueError as e:
            print(f"  Not enough data on {pitcher_name}'s arsenal to compare: {e}")
            return

    target_vector = candidate_vectors[pitcher_id]

    similar = rank_similar_pitchers(pitcher_id, target_vector, candidate_vectors, top_n=top_n)
    if similar.empty:
        print(f"  No comparable pitchers found yet for {pitcher_name}.")
        return

    names = get_pitcher_names(similar["pitcher_id"].tolist())
    similar["name"] = similar["pitcher_id"].map(names)

    rows = []
    for _, r in similar.iterrows():
        line = batter_vs_pitcher_line_from_pitches(cached["batter_pitches"], int(r["pitcher_id"]))
        rows.append({**r.to_dict(), **line})
    comps_with_stats = pd.DataFrame(rows)
    profile = weighted_batter_profile(comps_with_stats)

    print_scouting_summary(pitcher_name, batter_name, comps_with_stats, profile)


def live_mode(top_n: int = 5, poll_seconds: int = 20) -> None:
    game_pk = choose_live_game()
    print(f"\nTracking game {game_pk}. Checking every {poll_seconds}s for a new at-bat... (Ctrl+C to stop)")

    session_cache: dict = {}
    last_pair = None

    while True:
        matchup = get_current_matchup(game_pk)
        if matchup is None:
            print("  (no active at-bat right now)")
        else:
            pair = (matchup["batter_id"], matchup["pitcher_id"])
            if pair != last_pair:
                last_pair = pair
                print(f"\nNew at-bat: {matchup['batter_name']} vs {matchup['pitcher_name']}")
                run_scouting_for_matchup(
                    matchup["batter_id"], matchup["batter_name"],
                    matchup["pitcher_id"], matchup["pitcher_name"],
                    top_n, session_cache,
                )

        time.sleep(poll_seconds)


# MAIN

def main():
    # --- Mode toggle ---
    mode = input("Mode -- [1] single matchup lookup, [2] live game tracking: ").strip()
    if mode == "2":
        top_n = int(input("How many similar pitchers to compare against? ").strip())
        live_mode(top_n=top_n)
        return

    # --- Step 1: user input ---
    batter_name = input("Enter batter name: ").strip()
    pitcher_name = input("Enter pitcher name: ").strip()
    top_n = int(input("How many similar pitchers to compare against? ").strip())

    # --- Step 2: resolve IDs, get career range, get active faced pitchers ---
    batter_id = get_player_id(batter_name)
    target_pitcher_id = get_player_id(pitcher_name)

    start_dt, end_dt = get_batter_career_range(batter_id)
    arsenal_start_dt, arsenal_end_dt = get_arsenal_window(end_dt)
    print(f"\nPulling {batter_name}'s career: {start_dt} to {end_dt}...")

    # Fetch the batter's whole career ONCE, cached to disk -- reused below
    # for both the faced-pitcher list and every comp's stat line, instead
    # of re-downloading the same career data over and over.
    batter_pitches = get_cached_or_fetch_batter_pitches(batter_id, start_dt, end_dt)

    faced = get_all_faced_pitchers(batter_pitches)
    active_faced = filter_to_active_pitchers(faced)
    active_faced = rank_by_recency_and_cap(batter_pitches, active_faced)
    print(f"{len(active_faced)} active pitchers faced recently, capped/ranked by recency "
          f"(out of {len(faced)} total ever faced)")

    # --- Step 3: build the arsenal fingerprint matrix ---
    # Arsenals use a RECENT window (arsenal_start_dt/arsenal_end_dt), not the
    # batter's full career -- reflects current stuff and is far faster to fetch.
    print(f"\nFetching arsenal for target pitcher {pitcher_name} "
          f"(last ~{ARSENAL_LOOKBACK_DAYS // 365} seasons)...")
    try:
        target_vector = get_arsenal_with_fallback(target_pitcher_id, arsenal_start_dt, arsenal_end_dt, start_dt)
    except ValueError as e:
        print(f"\nCan't build a report: {pitcher_name} doesn't have enough MLB pitches on record yet ({e}).")
        return

    print(f"Fetching arsenals for {len(active_faced)} candidates in parallel...")
    candidate_vectors = fetch_arsenals_parallel(active_faced, arsenal_start_dt, arsenal_end_dt, start_dt)
    candidate_vectors[target_pitcher_id] = target_vector

    # --- Step 4: weighted Euclidean distance -> top N most similar ---
    similar = rank_similar_pitchers(target_pitcher_id, target_vector, candidate_vectors, top_n=top_n)
    names = get_pitcher_names(similar["pitcher_id"].tolist())
    similar["name"] = similar["pitcher_id"].map(names)

    # --- Step 5: batter's real history against each comp, blended ---
    print("\nSummarizing batter's history against each comp...")
    rows = []
    for _, r in similar.iterrows():
        line = batter_vs_pitcher_line_from_pitches(batter_pitches, int(r["pitcher_id"]))
        rows.append({**r.to_dict(), **line})
    comps_with_stats = pd.DataFrame(rows)
    profile = weighted_batter_profile(comps_with_stats)

    print_scouting_summary(pitcher_name, batter_name, comps_with_stats, profile)


if __name__ == "__main__":
    main()