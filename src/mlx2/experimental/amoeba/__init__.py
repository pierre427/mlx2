"""aMoEba mapping pipeline as one command (experimental, unqualified).

    mlx2 amoeba build --model <path> --out <dir>
    mlx2 amoeba status --out <dir>
    mlx2 amoeba resume --out <dir>
    mlx2 amoeba {inspect,plan} --model <path>
    mlx2 amoeba {map,core,boxes,partition,gates,bundle} --out <dir>

The procedure is docs/research/2026-11-13-amoeba-guide/mapping-workflow.md (Pierre, 2026-11-16,
the foundation document): six stages, the fixed data defaults of its section 4, the joint cone
carve of section 6, the +0.05-nats held verdict pooled over probe windows and 1 greedy : 3 T=0.7
continuations, the partition rules of step 4, automatic gating, and the section 10 bundle (tiles +
map + taxonomy version + prompt set Parquet/manifest + checkpoint fingerprint).

This package orchestrates. The carve itself, the map capture/cluster/label steps, the box-set
builder, the on-policy generation client and the prompt-set writer are the research code under
``research/amoeba/carve`` and ``scripts/amoeba_*`` (see ``PROVENANCE``); they run in child
processes, one model per process. Model-specific maths stays in the adapters and the carve host;
nothing here branches on a model name.
"""

# the core comes before the map: the map clusters the non-core units (mapping-workflow s5.1.2)
STAGES = ("core", "map", "boxes", "partition", "gates", "bundle")

PROVENANCE = {
    "source": "mlx2 research code, branch claude/amoeba-27b-80bc6a@52fd0d6c9 (FN stage scripts) and "
              "claude/amoeba-maps@c11aaf343 (map capture/cluster/label); same repository and licence",
    "reused_unchanged": [
        "research/amoeba/carve (python -m research.amoeba.carve: inspect, mean, rank, bisect, atlas)",
        "scripts/amoeba_topics_label.py (--core)", "scripts/amoeba_onpolicy_gen.py",
    ],
    "changed_here": {
        "scripts/amoeba_map_cluster.py": "--exclude (cluster the non-core units)",
        "scripts/amoeba_fn_boxset_build.py": "--topic (one box for one taxonomy id)",
        "scripts/amoeba_gen_stage.py": "SIGTERM stops its server; refuses a port already in use",
        "scripts/amoeba_prompt_set.py": "--capture-run (the map's probe windows)",
        "scripts/amoeba_topics_donor_gap.py": "--include-topic",
        "research/amoeba/carve/__main__.py": "--init-raw (a seed kept as given)",
    },
    "generalised": {
        "child_boxbatch.py": "scripts/amoeba_fn_boxbatch.py without the Flash-Next constants (PLE assert, 92 GB, expert "
                             "units come from the resource plan and the artifact)",
        "child_capture.py": "the topic-capture format (scripts/amoeba_topics_capture.py) on the base carve host, with "
                            "per-text activation peaks",
        "stages.run_partition": "the mapping-workflow step 4 table (scripts/amoeba_fn_box_partition.py adds FN-only "
                                "merges and folds the workflow does not have)",
    },
}
