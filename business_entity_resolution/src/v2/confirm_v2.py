"""ONE-SHOT confirmation on the sealed CONF fold (S1 id%5==1). Run once, after the recipe is frozen.
usage: python confirm_v2.py <baseline_run> <t_base> <challenger_run> <t_chal>
Paired bootstrap at family level: S1 entities sharing (country, core-name key) resample together."""
import json
import sys

import numpy as np
import polars as pl

from eval_v2 import EvalCtx, load_top3
from runlib import RUNS, cache_path, note
from scorer import paired_bootstrap

base, tb, chal, tc = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4])
lock = RUNS / "CONF_OPENED.json"
if lock.exists():
    raise SystemExit(f"CONF already opened: {lock.read_text()}  (a second look would make it development data)")
ctx = EvalCtx("CONF", "v2train")
da = ctx.score(load_top3(base, "v2train").filter(pl.col("p1") >= tb)).select("s1_idx", "country", "n_true", pl.col("F").alias("Fa"))
db = ctx.score(load_top3(chal, "v2train").filter(pl.col("p1") >= tc)).select("s1_idx", pl.col("F").alias("Fb"))
d = da.join(db, on="s1_idx")
fam = (pl.read_parquet(cache_path("v2train_source1_norm.parquet"), columns=["country", "name_core"]).with_row_index("s1_idx")
       .with_columns(pl.struct(pl.col("country"), pl.col("name_core").str.replace_all(" ", "")).hash(3).alias("fam")).select("s1_idx", "fam"))
d = d.join(fam, on="s1_idx")
mean, lo, hi = paired_bootstrap(d["Fa"].to_numpy(), d["Fb"].to_numpy(), d["fam"].to_numpy(), n=1000)
res = {"baseline": base, "t_base": tb, "challenger": chal, "t_chal": tc, "n_entities": d.height,
       "F_base": d["Fa"].mean(), "F_chal": d["Fb"].mean(), "delta": mean, "ci95": [lo, hi]}
for c in d["country"].unique().sort().to_list():
    s = d.filter(pl.col("country") == c)
    res[f"F_base_{c}"], res[f"F_chal_{c}"] = s["Fa"].mean(), s["Fb"].mean()
s0 = d.filter(pl.col("n_true") == 0)
res["F_base_zero_match"], res["F_chal_zero_match"] = s0["Fa"].mean(), s0["Fb"].mean()
lock.write_text(json.dumps(res, default=float))
print(json.dumps(res, indent=1, default=float))
note(f"CONFIRMATION (one-shot): base {res['F_base']:.5f} vs chal {res['F_chal']:.5f}, delta {mean:+.5f} CI95 [{lo:+.5f}, {hi:+.5f}]")
