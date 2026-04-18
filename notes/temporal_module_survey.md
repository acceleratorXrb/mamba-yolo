# Temporal Module Survey For Mamba-YOLO

## Downloaded references

- FGFA paper: `references/papers/FGFA_1703.10025.pdf`
- MEGA paper: `references/papers/MEGA_CVPR2020.pdf`
- LSFA paper: `references/papers/LSFA_2103.14529.pdf`
- SELSA paper: `references/papers/SELSA_1907.06390.pdf`

- FGFA code: `references/temporal_modules/fgfa`
- MEGA code: `references/temporal_modules/mega`
- LSFA code: `references/temporal_modules/lsfa`
- SELSA code: `references/temporal_modules/selsa`

## What existing VOD methods actually do

### FGFA
- Core idea: optical-flow-guided warping of nearby-frame features into the current frame, then weighted aggregation.
- Strength: explicit motion alignment.
- Cost: requires flow estimation and warping operators.
- Not a good first integration target for the current Ultralytics-style codebase because it adds a second heavy model path.

### MEGA
- Core idea: local + global memory bank aggregation for stronger temporal context.
- Strength: best semantics among classic VOD methods.
- Cost: complicated memory management, proposal-level aggregation, much heavier detector coupling.
- Too large for a first runnable graduation-project version.

### SELSA
- Core idea: sequence-level feature aggregation across proposal features.
- Strength: simple conceptual temporal semantics aggregation.
- Cost: depends on two-stage detector / RoI pipeline. Current Mamba-YOLO is one-stage anchor-free-ish YOLO style.
- Good conceptual reference, bad direct implementation match.

### LSFA
- Core idea: long/short-term feature aggregation in compressed video, emphasizing practical speed.
- Strength: lightweight temporal aggregation mindset.
- Cost: tied to key-frame / non-key-frame and compressed-video representations.
- The "short-term enhancement with previous context" idea is useful even if compressed-video specifics are not.

## Chosen first implementation

### Why this design
- The current codebase is single-frame YOLO + Mamba neck.
- The cheapest correct extension is not proposal memory or optical flow.
- The best first step is a previous-frame branch plus lightweight gated fusion at the three detect-input scales.

### Implemented temporal path
- Dataset now optionally returns:
  - `prev_img`
  - `prev_im_file`
  - `has_prev`
  - `sequence_id`
- Model now optionally runs:
  - current frame through backbone+neck
  - previous frame through the same backbone+neck
  - gated fusion on detect inputs `P3/P4/P5`
- Detect head stays unchanged.

### Why fuse at detect inputs
- It preserves the original Mamba-YOLO backbone/neck structure.
- It directly targets the three scales used for final detection.
- It keeps the first version small enough to debug and train.

## What this version is and is not

### This version is
- a real temporal model input path
- trainable with current YOLO loss
- suitable as the first runnable spatiotemporal baseline

### This version is not
- optical-flow aligned FGFA
- proposal-memory MEGA
- sequence-level RoI aggregation SELSA
- compressed-video LSFA

It is a pragmatic temporal extension that fits the current repository.
