# Model bench harness

Harness used for the 2026-08-01 guided-decoding experiment (#181, #203). Kept
because it generalises: it answers "does model/config change X affect answer
quality, terminology, tool flows or latency?" for any A/B on this stack.

Voice-side runners live in `voice-oan-api/scripts/` — they import voice agent
modules, so they cannot run from this repo.

## Files

| file | what it does |
|---|---|
| `run_bench.py` | Runs questions through the agent directly (no FastAPI/Redis/JWT). Optional vLLM guided decoding. |
| `run_bench_parity.py` | `run_bench.py` + `--inject-rules` (terminology rules into the agent prompt) and `--apply-map` (deterministic forbidden-term replacement). |
| `run_bench_sandwich.py` | Prod-equivalent arm — drives the real `stream_chat_messages`, so pre/post-translation and moderation all run. |
| `guided_patch.py` | Port of `bharat-oan-api#142`. Builds the script-constraining regex. `--strict-142` reproduces the original char set. |
| `rubric_score.py` | Mechanical scoring: script drift, forbidden terms, self-reference gender, placeholders, markdown, latency, tool mix. |
| `report.py` | Side-by-side table for N arms. |
| `judge.py` | Blind pairwise LLM-as-judge + faithfulness against retrieved context. |
| `posthoc_map.py` | Applies the forbidden map to existing results — measures what the deterministic layer would fix. |
| `gender_check.py` | Coverage of `GU_FEMININE_SELF_REFERENCE_REPLACEMENTS` against actual output. |

## Running

Runners must run with **cwd = repo root** — assets load via relative paths.

```bash
python scripts/run_bench.py --questions q.csv --out results.csv --arm control
python scripts/run_bench.py --questions q.csv --out guided.csv --arm guided --guided
python scripts/report.py assets/gu_term_policy.json q.csv results.csv guided.csv
```

Question CSV needs `id,question,lang`. `lang` is honoured per row.

## Things that will bite you

- **Use `structured_outputs`, never `guided_regex`.** On vLLM 0.15.1 the legacy
  key is silently ignored — no error, just an unconstrained run that looks clean.
- **Always run a same-day paired control.** Do not compare against an older
  baseline; identical unguided code measured 15.17s mean in May and 28.69s in
  August. Run arms concurrently so both see the same GPU load.
- **Randomize A/B slots in `judge.py`.** Judge position bias exceeded the arm
  effect in our run (slot B preferred 53:32 in one pair, reversed 188:167 in
  another). A fixed layout would have swamped the signal.
- **Score drift against each row's own expected script.** Scoring a Hindi row
  against Gujarati counts correct output as drift.
- **Guided decoding degrades silently.** If the model must emit a string outside
  the allowed character class (a fixed refusal in another script, an internal
  structured output) it produces garbage at HTTP 200 rather than erroring.
- Check Redis is up if any tool under test reads it, or tool metrics will
  understate real usage.

## Results

Full writeup and raw CSVs are not in this repo. See #181 and the report
`bench-2026-08-01-guided-decoding.md`.
