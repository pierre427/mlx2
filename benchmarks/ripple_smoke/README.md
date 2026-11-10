# Ripple serving smoke filler

`filler.md` is the varied context `scripts/probe_ripple_serving.py` puts before the
`ZX-3187` identifier question (prompt version `varied-v1`, the default). It is a frozen
copy of the first 3027 characters of `docs/QUALIFICATION.md` at commit
5557932568b9b7f0d65b444c478af939fc1ce87b, so the prompt does not drift when the docs change.
sha256 `30db3f12a62b2491b547ac2ec2deddb2220e944f976fe544139ebfcfc4caedcc`.

It replaced `"Inventory checked. " * 160` (2026-11-06, sweep item RIPPLE-4): reasoning models
do badly on repeated-string input, so that shape can fail a smoke on its own. The old prompt is
still available as `--prompt legacy-repeated` to reproduce the 2026-11-04 receipts exactly.
