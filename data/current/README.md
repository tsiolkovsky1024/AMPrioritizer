# v2 censor-safe data

Canonical protocol: `canonical_mmseqs80_v2_censor_safe`.

Repeated exact measurements use a geometric mean. Repeated right-censored
measurements retain the maximum lower bound; repeated left-censored
measurements retain the minimum upper bound. Groups mixing observation types
are excluded because the current row schema cannot represent their joint
likelihood without information loss. The top-20 species order and MMseqs2 80%
cluster split are frozen.

Use `amp-prioritizer data verify --version v2_censor_safe` before any formal
experiment.

