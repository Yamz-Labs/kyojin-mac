"""CPU tests of tools/qualify (planner, rules, search, pack metadata, report). Run: cd tools/qualify && python3 -m unittest discover -s tests (no weights, no GPU)"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pack as packmod  # noqa: E402
import plan as planmod  # noqa: E402
import report  # noqa: E402
import rules  # noqa: E402
import search  # noqa: E402
import stages  # noqa: E402


def fake_pack(d: Path, arch="Qwen3_5ForConditionalGeneration", mtp=0, template=True, quant="exl3", extra=None):
    t = {"model_type": "qwen3_5_text", "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16, "hidden_size": 128,
         "max_position_embeddings": 65536, "vocab_size": 100, **(extra or {})}
    if mtp:
        t["mtp_num_hidden_layers"] = mtp
    (d / "config.json").write_text(json.dumps({"architectures": [arch], "model_type": "qwen3_5", "text_config": t}))
    (d / "quantization_config.json").write_text(json.dumps({"quant_method": quant, "bits": 4.0, "head_bits": 6}))
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 100)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}))
    if template:
        (d / "chat_template.jinja").write_text("{{ messages }}")
    return d


def args(**kw):
    a = dict(backend="mlx", scale="air", sessions=[1, 4], ref=None, kv_kl_max=0.02, kv_bits=[8, 6, 4], ctx_cap=None, out="/tmp/x", doc_tokens=512, kv_n=2048,
             kv_eval=128, mem_cap_gib=13.0)
    a.update(kw)
    return a


class PackTests(unittest.TestCase):
    def test_hybrid_flags_come_from_the_pack(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t)))
            self.assertEqual((m["family"], m["kv_mode"], m["kv_layers"], m["kv_heads"], m["head_dim"]), ("hybrid_gdn", "kvwin", 4, 2, 16))
            f = packmod.serve_flags(m, port=1, ctx=4096, sessions=2, k_bits=6, v_bits=6, chunk=512)
            self.assertEqual(f[:3], ["-m", "kyojin_mlx.serve.server", str(Path(t))])
            self.assertEqual(f[f.index("--ctx") + 1], "4096")
            self.assertNotIn("--mtp-depth", f)                                  # no MTP head
            self.assertEqual(m["template"], "chat_template.jinja")
            self.assertEqual(m["problems"], [])

    def test_kv_choice_is_the_kvwin_environment(self):
        self.assertEqual(packmod.kvwin_env(6, 6), {"KYOJIN_KVWIN": "6:6:64:4:128"})
        self.assertEqual(packmod.kvwin_env(16, 16), {})

    def test_mtp_head_selects_depth(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t), mtp=3))
            self.assertEqual(packmod.mtp_depth(m), 0)                                # the Qwen server path has no speculative decode
            m = packmod.read_pack(fake_pack(Path(t), arch="DeepseekV41ForCausalLM", mtp=3))
            self.assertEqual(packmod.mtp_depth(m), 3)
            self.assertIn("--mtp-depth", packmod.serve_flags(m, port=1, ctx=4096, sessions=2, k_bits=6, v_bits=6))

    def test_ds41_has_a_native_cache_and_no_kv_formula(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t), arch="DeepseekV41ForCausalLM", template=False))
            self.assertEqual((m["family"], m["kv_mode"], m["template"]), ("ds41", "native", "builtin:ds41_chat"))
            self.assertIsNone(packmod.kv_bytes_per_token(m, 6, 6))
            self.assertEqual(packmod.serve_env(m, 16, 16), {})

    def test_template_in_tokenizer_config_counts(self):
        with tempfile.TemporaryDirectory() as t:
            d = fake_pack(Path(t), template=False)
            self.assertIn("no chat template", packmod.read_pack(d)["problems"][0])
            (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
            self.assertEqual(packmod.read_pack(d)["problems"], [])

    def test_wrong_quant_is_reported(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t), quant="gptq"))
            self.assertEqual(len(m["problems"]), 1)

    def test_hybrid_gdn_counts_only_full_attention_layers(self):
        t = {"num_hidden_layers": 64, "full_attention_interval": 4}
        self.assertEqual(packmod.kv_layers(t), 16)
        self.assertEqual(packmod.kv_layers({"num_hidden_layers": 4, "layer_types": ["linear_attention", "full_attention"] * 2}), 2)

    def test_kv_bytes_per_token_of_the_affine_pool(self):
        # Qwen3.5-0.8B: 6 full-attention layers x 2 KV heads x 256: k6v6 with a fp16 scale + bias per group of 64 = 6.5 bits per element
        m = dict(kv_mode="kvwin", kv_layers=6, kv_heads=2, head_dim=256)
        self.assertAlmostEqual(packmod.kv_bytes_per_token(m, 6, 6), 6 * 2 * 256 * 13 / 8)
        self.assertLess(packmod.kv_bytes_per_token(m, 6, 6), packmod.kv_bytes_per_token(m, 8, 8))

    def test_verify_shards(self):
        with tempfile.TemporaryDirectory() as t:
            d = fake_pack(Path(t))
            self.assertEqual(packmod.verify_shards(d), [])
            (d / "model-00001-of-00001.safetensors").unlink()
            self.assertEqual(packmod.verify_shards(d), ["missing shard model-00001-of-00001.safetensors"])

    def test_single_file_pack_without_index(self):
        with tempfile.TemporaryDirectory() as t:
            d = fake_pack(Path(t)); (d / "model.safetensors.index.json").unlink()
            self.assertEqual(packmod.verify_shards(d), [])
            self.assertFalse(packmod.read_pack(d)["has_index"])


class PlanTests(unittest.TestCase):
    def meta(self, **kw):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t), **kw))
        return m

    def names(self, plan, status):
        return [s.name for s in plan if s.status == status]

    def test_full_plan_without_ref_skips_only_parity(self):
        p = planmod.build_plan(self.meta(), args(), {})
        self.assertEqual([s.name for s in p], planmod.ORDER)
        self.assertEqual(self.names(p, "skip"), ["parity"])                      # kv is measured against the pack's own fp16 cache: no --ref needed

    def test_ref_enables_parity_for_a_family_with_an_adapter(self):
        p = planmod.build_plan(self.meta(), args(ref="/r"), {})
        self.assertEqual(self.names(p, "skip"), [])

    def test_family_without_adapter_is_unmeasured_not_fatal(self):
        p = planmod.build_plan(self.meta(arch="DeepseekV41ForCausalLM"), args(ref="/r"), {})
        self.assertIn("no reference adapter", next(s for s in p if s.name == "parity").note)
        self.assertIn("native cache", next(s for s in p if s.name == "kv").note)

    def test_reference_larger_than_the_cap_is_skipped(self):
        m = self.meta(); m["ref_bytes"] = 40 * 2**30
        p = planmod.build_plan(m, args(ref="/r"), {})
        self.assertIn("does not fit", next(s for s in p if s.name == "parity").note)
        m["ref_bytes"] = 2 * 2**30
        self.assertEqual(self.names(planmod.build_plan(m, args(ref="/r"), {}), "skip"), [])

    def test_resume_skips_done_stage_with_same_key_only(self):
        m = self.meta()
        key = planmod.fingerprint("maxctx", m, args())
        st = {"maxctx": dict(status="done", key=key, wall_s=12.0, data={})}
        self.assertEqual(self.names(planmod.build_plan(m, args(), st), "done"), ["maxctx"])
        self.assertEqual(self.names(planmod.build_plan(m, args(sessions=[1]), st), "done"), [])      # changed input -> stale
        km = dict(status="done", key=planmod.fingerprint("meta", m, args()), wall_s=0, data={})
        self.assertEqual(self.names(planmod.build_plan(m, args(sessions=[1]), {"meta": km}), "done"), ["meta"])   # unrelated arg keeps meta
        self.assertEqual(self.names(planmod.build_plan(m, args(), st, redo=["maxctx"]), "done"), [])

    def test_stage_selector_blocks_missing_dependency(self):
        p = planmod.build_plan(self.meta(), args(), {}, only=["serve"])
        self.assertEqual(self.names(p, "blocked"), ["serve"])
        p = planmod.build_plan(self.meta(), args(), {}, only=["serve"], force_deps=True)
        self.assertEqual(self.names(p, "blocked"), [])
        self.assertIn("maxctx", self.names(p, "todo"))

    def test_stage_selector_runs_cpu_stages_alone(self):
        p = planmod.build_plan(self.meta(), args(), {}, only=["fetch", "meta"])
        self.assertEqual(self.names(p, "todo"), ["fetch", "meta"])
        self.assertEqual(planmod.estimate(p)["gpu_s"], 0)

    def test_estimates_follow_the_scale(self):
        air = planmod.estimate(planmod.build_plan(self.meta(), args(scale="air"), {}))["total_s"]
        ultra = planmod.estimate(planmod.build_plan(self.meta(), args(scale="ultra"), {}))["total_s"]
        self.assertLess(air, ultra)

    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            planmod.save_stage(Path(t), "meta", "k", "done", 1.23, {"a": 1})
            self.assertEqual(planmod.load_state(Path(t))["meta"]["data"], {"a": 1})


class RuleTests(unittest.TestCase):
    def test_kv_choice_follows_drill0_numbers(self):
        self.assertEqual(rules.choose_kv({6: 0.014, 4: 0.123}), (6, 6))          # Nanbeige: k6v6 vs k8v8 KL .014 -> Q6
        self.assertEqual(rules.choose_kv({6: 0.05}), (8, 8))                     # k6v6 too lossy -> k8v8
        self.assertEqual(rules.choose_kv({6: 0.05, 7: 0.01}), (7, 7))
        self.assertEqual(rules.choose_kv({4: 0.001, 3: 0.002}), (8, 8))          # never below Q6 even if Q4 looks fine
        self.assertEqual(rules.choose_kv({}), (8, 8))

    def good(self):
        return dict(parity=dict(kl_mean=0.086, by_domain={"wiki": dict(top1=0.79)}), kv=dict(chosen=[6, 6]), maxctx={"1": 77824, "4": 40960},
                    serve=dict(sha_self_equal=True, oom=False, conformance=dict(ok=True)), smoke=dict(passed=4, total=4))

    def test_verdict(self):
        self.assertTrue(rules.verdict(self.good())["pass_"])
        q = self.good(); q["parity"]["by_domain"]["wiki"]["top1"] = 0.5
        self.assertIn("wiki", rules.verdict(q)["reasons"][0])
        q = self.good(); q["serve"]["sha_self_equal"] = False
        self.assertFalse(rules.verdict(q)["pass_"])
        q = self.good(); q["smoke"] = dict(passed=1, total=4)
        self.assertFalse(rules.verdict(q)["pass_"])

    def test_unmeasured_is_never_a_pass(self):
        q = self.good(); q["parity"] = None
        v = rules.verdict(q)
        self.assertFalse(v["pass_"]); self.assertEqual(v["unmeasured"], ["parity"])


class SearchTests(unittest.TestCase):
    def run_search(self, limit, lo=8192, hi=131072):
        calls = []

        def probe(c):
            calls.append(c)
            return c <= limit, "x"
        return search.bisect_max(lo, hi, probe), calls

    def test_finds_grid_value_below_limit(self):
        r, calls = self.run_search(77_900)
        self.assertEqual(r["best"], 77_824)
        self.assertLessEqual(len(calls), 8)
        self.assertTrue(all(c % search.STEP == 0 for c in calls))

    def test_upper_bound_when_everything_fits(self):
        r, calls = self.run_search(10**9)
        self.assertEqual((r["best"], calls), (131072, [131072]))

    def test_nothing_fits(self):
        r, _ = self.run_search(1000)
        self.assertIsNone(r["best"])

    def test_upper_bound_from_memory(self):
        ub = search.upper_bound(free_bytes=6.0e9, bytes_per_token=57_000, headroom_bytes=2.5 * 2**30, cap=524288)
        self.assertEqual(ub % search.STEP, 0)
        self.assertEqual(search.upper_bound(1e12, 1000, 0, 65536), 65536)


class ParityTests(unittest.TestCase):
    def test_conversion_report_summary(self):
        with tempfile.TemporaryDirectory() as t:
            rows = [dict(layer=0, name="mlp.up_proj", K=2, proxy=0.04, g=0.7), dict(layer=0, name="attn.q", K=3, proxy=0.006, g=0.6),
                    dict(layer=1, name="mlp.up_proj", K=2, proxy=0.06, g=0.7)]
            (Path(t) / "conv_report.json").write_text(json.dumps(rows))
            s = stages.conv_summary(Path(t), None)
            self.assertTrue(s["measured"]); self.assertEqual(s["worst_layer"], 1); self.assertEqual(s["bits_hist"], {"2": 2, "3": 1})
            self.assertFalse(stages.conv_summary(Path(t) / "none", None)["measured"])

    def test_aggregate_is_token_weighted_and_ignores_meta_keys(self):
        r = stages.aggregate_parity({"a": dict(kl=0.1, top1=0.9, dnll=0.0, tokens=100), "b": dict(kl=0.3, top1=0.7, dnll=0.1, tokens=300), "_peak_gib": 1.0})
        self.assertAlmostEqual(r["kl_mean"], 0.25)


class ReportTests(unittest.TestCase):
    def test_assemble_and_render(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t)))
        m["draft_policy"] = "off"
        done = lambda d: dict(status="done", wall_s=1.0, data=d)
        stg = {"maxctx": done({"1": 77824, "4": 40960, "tried_s1": []}),
               "kv": done(dict(chosen=[6, 6], kl_max=0.02, rule="r", table={"8": dict(kl_ref=0.04, top1_ref=0.9), "6": dict(kl_ref=0.05, top1_ref=0.89, kl_vs_k8=0.014)})),
               "serve": done(dict(cmd=["py", "serve.py"], sha_self_equal=True, oom=False, conformance=dict(ok=True), decode_tps_median=36.7, prefill_8k_tps=1380.0,
                                  curve=[dict(prompt=2048, ttft=1.5, prefill_tps=1365.0, dec_tps=37.0, err=None)], agg=[dict(n=1, agg_dec_tps=36.0, dec_tps=[36.0], ttft=[5.8])],
                                  sha=[dict(sha="abc")])),
               "smoke": done(dict(kind="realuse_b", total=3, passed=3, checks=[dict(name="completed", ok=True)]))}
        j = report.assemble(m, args(), stg, HERE)
        self.assertEqual(j["serve"]["ctx"], {"1": 77824, "4": 40960})
        self.assertEqual(j["serve"]["kv"], [6, 6])
        self.assertEqual(j["verdict"]["unmeasured"], ["parity"])
        md = report.render_md(j)
        self.assertIn("max -c", md); self.assertIn("36.7", md); self.assertIn("generic", md)
        json.dumps(j, default=str)


class SmokeTests(unittest.TestCase):
    def test_judge_reads_realuse_b_json(self):
        import smoke
        sub = dict(tokens=32, tps=30.0)
        j = dict(alone_main=dict(tokens=64), spawn=dict(subs=[sub, sub], agg_subs_tps=58.0, main_max_gap_after_spawn=0.4), agree_vs_solo=[1.0, 0.97],
                 turns=[dict(agree=[1.0, 1.0])], solo_tps=31.0)
        r = smoke.judge(j)
        self.assertEqual((r["passed"], r["total"]), (3, 3))
        j["spawn"]["agg_subs_tps"] = 20.0                        # slower than one agent alone
        self.assertEqual(smoke.judge(j)["passed"], 2)
        j["agree_vs_solo"] = [0.2, 1.0]
        self.assertEqual(smoke.judge(j)["passed"], 1)


class CliTests(unittest.TestCase):
    def test_dry_run_touches_nothing_and_prints_plan(self):
        import io, contextlib, qualify
        with tempfile.TemporaryDirectory() as t:
            pk = fake_pack(Path(t) / "p") if (Path(t) / "p").mkdir() is None else None
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = qualify.main(["--backend", "mlx", "--scale", "air", "--pack", str(pk), "--dry-run", "--out", str(Path(t) / "o")])
            self.assertEqual(rc, 0)
            self.assertIn("serve command", buf.getvalue())
            self.assertEqual([p.name for p in (Path(t) / "o").iterdir()], [])


if __name__ == "__main__":
    unittest.main()
