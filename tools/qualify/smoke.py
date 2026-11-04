"""Real-use smoke on the Mac: DAY1 real-use scenario B (macair43's tools/realuse_b.py): a main agent + N sub-agents of the same model sharing a long
prefix, then tool-style turns, black box over /v1/completions on the SAME server the serve stage measured. Checks (each one a pass / fail, all
configurable through the numbers they report):
  1. completed      the tool exited 0 and every request got tokens (no error, no empty stream)
  2. outputs agree  every sub-agent's concurrent output matches its solo run on the hot prefix (equal, or agreement before the first difference >= 0.9;
                    bitwise with KYOJIN_ROWS_NAX=0, a fp16 rounding class with NAX on)
  3. multi-user     the sub-agents' aggregate decode >= solo (the owner's 'solo never slower' / aggregate(N) close to N x solo: the ratio is reported)
The CUDA/ROCm versions drive the Hermes CLI (hermes_opt/eval); the Hermes CLI is not installed on the Mac, so the Hermes tasks stay a day-1 item on the Ultra
(DAY1_RUNBOOK). The scripted tool loop of the CUDA/ROCm smoke needs a tool-calling model: not meaningful on a Base pack."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def realuse_cmd(py: str, repo: Path, port: int, subs: int, scale: dict, vocab: int | None, out: Path) -> list:
    c = [py, "-u", str(repo / "tools/realuse_b.py"), "--port", str(port), "--prefix", str(scale["prefix"]), "--task", str(scale["task"]), "--decode", str(scale["decode"]),
         "--sub", str(scale["sub"]), "--turns", str(scale["turns"]), "--arms", "spawn", "--subs", str(subs), "--json", str(out)]
    return c + (["--vocab", str(vocab)] if vocab and vocab <= 8192 else ["--text"])


def judge(j: dict) -> dict:
    """The three checks from realuse_b's JSON."""
    subs = (j.get("spawn") or {}).get("subs") or []
    done = bool(j.get("alone_main", {}).get("tokens")) and bool(subs) and all(s.get("tokens") for s in subs)
    agree = j.get("agree_vs_solo") or []
    turns = [a for t in j.get("turns", []) for a in t.get("agree", [])]
    ag = min(agree + turns) if (agree or turns) else None
    agg, solo = (j.get("spawn") or {}).get("agg_subs_tps"), j.get("solo_tps")
    ratio = (agg / solo) if agg and solo else None
    checks = [dict(name="completed", ok=done),
              dict(name="outputs_agree", ok=ag is not None and ag >= 0.9, value=ag),
              dict(name="multiuser_not_slower", ok=ratio is not None and ratio >= 1.0, value=ratio, agg_tps=agg, solo_tps=solo, subs=len(subs))]
    return dict(kind="realuse_b", total=len(checks), passed=sum(c["ok"] for c in checks), checks=checks,
                main_max_gap_after_spawn=(j.get("spawn") or {}).get("main_max_gap_after_spawn"), subs=len(subs))


def run_realuse(py: str, repo: Path, env: dict, port: int, subs: int, scale: dict, vocab: int | None, out: Path, timeout: int = 7200) -> dict:
    import os
    cmd = realuse_cmd(py, repo, port, subs, scale, vocab, out)
    r = subprocess.run(cmd, cwd=repo, env={**os.environ, **env}, capture_output=True, text=True, timeout=timeout)
    out.with_suffix(".log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
    if r.returncode or not out.exists():
        return dict(kind="realuse_b", total=3, passed=0, error=(r.stderr or r.stdout)[-400:], checks=[])
    return judge(json.loads(out.read_text()))
