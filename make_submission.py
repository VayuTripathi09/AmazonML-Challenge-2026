"""Build <team_name>_submission.zip in the layout required by the challenge:

<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/*.py, README.md, requirements.txt}
└── Documentation_template.md

usage: python make_submission.py <team_name>
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(team: str):
    out = ROOT / f"{team}_submission.zip"
    files = {
        ROOT / "output" / "matching_results.tsv": "output/matching_results.tsv",
        ROOT / "output" / "candidate_pairs.tsv": "output/candidate_pairs.tsv",
        ROOT / "Documentation_template.md": "Documentation_template.md",
        ROOT / "business_entity_resolution" / "README.md": "code/business_entity_resolution/README.md",
        ROOT / "business_entity_resolution" / "requirements.txt": "code/business_entity_resolution/requirements.txt",
    }
    src = ROOT / "business_entity_resolution" / "src"
    for py in sorted(src.rglob("*.py")):  # includes src/v2/
        files[py] = f"code/business_entity_resolution/src/{py.relative_to(src).as_posix()}"
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        sys.exit(f"missing files: {missing}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files.items():
            z.write(src, arc)
            print(f"  + {arc}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "team")
