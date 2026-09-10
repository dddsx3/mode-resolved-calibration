# Data sources

This repository **does not contain any third-party raw data**.  The frozen numerical
artifacts under `results/` and the metadata under `data/manifests/` are self-contained;
raw downloads must be obtained upstream.

## OpenIllumination

- Source: `OpenIllumination/OpenIllumination` on the Hugging Face Hub (CC BY 4.0).
- Download all capturings with your usual HF client, e.g.
  `huggingface-cli download OpenIllumination/OpenIllumination --local-dir D:/data/OpenIllumination --repo-type dataset`
- Used layers: for each object, the masked thumbnails
  `OLAT/<obj>/Lights/<NNN>/com_masked_thumbnail/<CAM>.png` (200×273 RGBA, masked,
  black background) and masks `output/com_masks/<CAM>.png` (camera `A1`);
  ground-truth lights from the root `light_pos.npy` (142 lights).
  The full-resolution stack is not needed; the per-object selection is frozen/recorded
  in `data/manifests/openillumination_dev_manifest.json` (with per-file SHA-256).
- Held-out objects used here (11): `obj_03_pumpkin, obj_04_dolphin, obj_07_pumpkin2,
  obj_09_ball, obj_10_pumpkin3, obj_11_pine, obj_13_mushroom, obj_16_friends_cup,
  obj_17_pumpkin5, obj_18_fabric_hat, obj_19_cylinder`
  (`obj_20_greenhead` lacks the thumbnail layer on the HF side).

## DiLiGenT

- Source: official DiLiGenT MV / photometric-stereo dataset distribution.
- Used: `pmsData/<object>` folders, 96 lights per object; the surface mask via the
  provided geometry/normal maps; ground-truth normals used for evaluation only.
- Loading contract: `src/calibinfo/datasets/diligent.py` (shapes, mask trimming,
  per-file SHA-256 manifest).

## Corruption protocol (OpenIllumination, joint)

- Corruption: jointly re-scale intensity and the direction-position coordinates.
- Levels: `[0.1, 0.2, 0.35, 0.5, 0.75, 1.0]` (0.1 = smallest). Seeds: 20 per
  (object, level) cell. Every run adds the corrupted render to the empirical column;
  predicted damage is computed deterministically from the retained spectrum.

## Reproducing data-dependent steps

Only §5–§7 of `docs/EXPERIMENTS.md` (OpenIllumination validation/severity, DiLiGenT
sanity) need raw data; the deterministic checks (identity panels, gauge spectrum,
Monte-Carlo grids) and the full test suite run without any download. For the
data-dependent parts, supply the paths through
`configs/openillumination.yaml` (`data_root`, `data_meta`) and
`configs/diligent.yaml` (`data_root`).