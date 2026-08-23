# Continuation / partial-restore diagnostic (SUPPORT_ONLY)

NOVELTY STATUS: CLASSICAL COUPLED/NONRECTANGULAR ROBUST ADVANTAGE AND
CROSS-WORLD PARTIAL IDENTIFICATION EQUIVALENCE (design 3.4, 13.4)
CLAIM STATUS: SUPPORT_ONLY — formally sound; not a new theory.

## U1 partial restore (observation regimes)
- marginal mixed fibers: [(0,), (1,)]
- paired-replay mixed fibers: []
- paired-replay identifiable fibers: [(0, 0), (0, 1), (1, 0), (1, 1)]
- lesson: marginal fibers mix signs -> abstain; paired replay identifies the replay summary, not the original-state same-noise effect sign without a bridge assumption (design 8.4).

## U2/U3 continuation family
- Q(0,0) values over the family: [0.5403, 0.5176, 0.4949]
- Q(0,1) values over the family: [0.5125, 0.5083, 0.5042]
- sign stability: {'Q0': ['positive', 3], 'Q1': ['positive', 3]}
- rank reversals: 1
- box vs coupled: {'box_size': 9, 'coupled_size': 3, 'nonrectangular': True}
- lesson: sign/rank stability is family-relative; the coordinate box is larger than the coupled realization set when the family is nonrectangular.

## Honesty notes
- docs_only_semantic: new frozen universe; legacy counts (96 rows, 5/36, 100/749, 400/120000 reversals) are incident background.
- The historical finding 'replica cannot identify original-state same-noise effect sign without a bridge assumption' is reproduced as a TYPE (marginal mixed fibers).