# The engine with every earlier language

`embodied_lot.py` here is the engine as of tag `all-languages` (28 Oct 2026). Every change to the language up to
then is a switch (`PURE_CLAIMS`, `STATIC_FIELDS`, `STANDING_MOTION`, ..., `MISSIONS`, `SEE_START`, and the `legacy`
dynamics), so an earlier run's semantics can be set again. The engine at the repository root has only the current
language built in. `demos.py` and `visualize_demo.py` are the original worked examples (the `legacy` step).

To rerun an earlier fit or analysis exactly, check out the tag, where `grace_lot/` still reads the switches from a
run's config:

    git worktree add ../elot-all-languages all-languages

To run the old demos from the repository root:

    PYTHONPATH="legacy/all_languages;." python legacy/all_languages/demos.py
