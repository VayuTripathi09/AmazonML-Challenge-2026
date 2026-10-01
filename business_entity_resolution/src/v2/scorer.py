"""Official macro-per-S1 F0.5 on integer ids.

Per S1 entity: F0.5 = 5*TP / (5*TP + 4*FP + FN); empty truth & empty prediction -> 1.0,
empty truth & any prediction -> 0.0. Every S1 in `eval_s1` is averaged (singletons included).
"""
import numpy as np
import polars as pl


def per_entity(pred: pl.DataFrame, truth: pl.DataFrame, eval_s1: pl.DataFrame) -> pl.DataFrame:
    """pred/truth: (s1_idx, q_idx) pairs; eval_s1: (s1_idx, ...). Returns eval_s1 + TP, FP, FN, F."""
    ev = eval_s1.select(pl.col("s1_idx").cast(pl.UInt32), *[c for c in eval_s1.columns if c != "s1_idx"])
    p = pred.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("q_idx").cast(pl.UInt32)).unique()
    t = truth.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("q_idx").cast(pl.UInt32)).unique()
    p = p.join(ev.select("s1_idx"), on="s1_idx", how="semi")
    t = t.join(ev.select("s1_idx"), on="s1_idx", how="semi")
    tp = p.join(t, on=["s1_idx", "q_idx"], how="semi").group_by("s1_idx").agg(pl.len().alias("TP"))
    npred = p.group_by("s1_idx").agg(pl.len().alias("NP"))
    ntrue = t.group_by("s1_idx").agg(pl.len().alias("NT"))
    d = ev.join(tp, on="s1_idx", how="left").join(npred, on="s1_idx", how="left").join(ntrue, on="s1_idx", how="left")
    d = d.with_columns(pl.col("TP", "NP", "NT").fill_null(0).cast(pl.Int64)).with_columns(
        (pl.col("NP") - pl.col("TP")).alias("FP"), (pl.col("NT") - pl.col("TP")).alias("FN"))
    denom = 5 * pl.col("TP") + 4 * pl.col("FP") + pl.col("FN")
    return d.with_columns(pl.when(denom == 0).then(1.0).otherwise(5 * pl.col("TP") / denom).alias("F"))


def macro(pred, truth, eval_s1) -> float:
    return per_entity(pred, truth, eval_s1)["F"].mean()


def paired_bootstrap(fa: np.ndarray, fb: np.ndarray, groups: np.ndarray, n=1000, seed=0):
    """Entity/family-level paired bootstrap of mean(fb - fa). groups: family id per entity (resampled as units)."""
    rng = np.random.default_rng(seed)
    diff = fb - fa
    uniq, inv = np.unique(groups, return_inverse=True)
    gsum = np.bincount(inv, weights=diff)
    gcnt = np.bincount(inv)
    stats = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(uniq), len(uniq))
        stats[i] = gsum[idx].sum() / gcnt[idx].sum()
    return diff.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def _tests():
    E = pl.DataFrame({"s1_idx": [0, 1, 2, 3, 4]})
    T = pl.DataFrame({"s1_idx": [0, 0, 1, 1, 3], "q_idx": [10, 11, 20, 21, 40]})
    # 0: perfect; 1: one miss; 2: empty truth, empty pred; 3: false merge + hit; 4: empty truth, false match
    P = pl.DataFrame({"s1_idx": [0, 0, 1, 3, 3, 4], "q_idx": [10, 11, 20, 40, 99, 50]})
    d = per_entity(P, T, E).sort("s1_idx")
    exp = [1.0, 5 / (5 + 0 + 1), 1.0, 5 / (5 + 4 + 0), 0.0]
    assert np.allclose(d["F"].to_numpy(), exp), d
    # formula equals (1.25 P R)/(0.25 P + R) for entity 3: P=0.5,R=1 -> 0.625/1.125
    assert abs(exp[3] - 1.25 * 0.5 * 1 / (0.25 * 0.5 + 1)) < 1e-12
    # missed-only entity: pred empty, truth non-empty -> 0
    assert per_entity(P.filter(pl.col("s1_idx") != 1), T, E).sort("s1_idx")["F"][1] == 0.0
    # mixed sizes / duplicates in pred ignored
    assert macro(pl.concat([P, P]), T, E) == macro(P, T, E)
    print("scorer tests passed:", exp)


if __name__ == "__main__":
    _tests()
