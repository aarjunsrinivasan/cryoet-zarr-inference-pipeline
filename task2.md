# AI Engineer: AI+CryoET Technical Task

## Introduction

Cryo-electron tomography (cryoET) produces 3D reconstructions of cellular environments at near-molecular resolution. Particle picking—detecting and localizing biomolecules in tomograms—is a core problem, in particular in crowded cellular environments. This task explores particle picking on **synthetic and experimental in-situ cellular tomograms**.

### Resources

- **Model: TopCUP** (1st-place CZII Kaggle solution)
  - Repo: <https://github.com/czimaginginstitute/czii_cryoet_mlchallenge_winning_models>
  - Tutorial: <https://virtualcellmodels.cziscience.com/tutorial/topcup-tutorial>

- **Data utilities**
  - CZ cryoET Data Portal client: <https://chanzuckerberg.github.io/cryoet-data-portal/stable/>
  - `copick` (data abstraction used by TopCUP): <https://github.com/copick/copick>
- **Stack**: PyTorch + MONAI, Python 3.11 recommended. A GPU is recommended but not required. See notes below for details.

### Environment heads-up

TopCUP 1.2.2 pins `numcodecs==0.11.0` as a dependency, which causes multiple problems. Use the following approaches to resolve the problems on different platforms:
- **Linux/Windows**: Use Python 3.11.
- **Apple Silicon**: Use python 3.12 and _override_ `numcodecs` to `>=0.12,<0.13`. See your environment manager's documentation for how to properly override dependencies.

### Instructions

- All data and model downloads must be programmatic—no manual downloads.
- Make sure you understand the problem this task is trying to solve, the approach TopCUP is using to solve it, and provide reasoning for your own choices and suggestions.
- If you have questions, please reach out to [Michael Innerberger](mailto:innerbergerm@hhmi.org) or [Magdalena Schneider](mailto:schneiderm2@hhmi.org).

A typical candidate proficient in Pytorch and familiar with basic concepts of CryoET will likely spend around 4 hours on this task. Time is not an evaluation criterion and this estimate should help you keep the task manageable.
On top of this, plan for 1-2h of inference time if you are running the task on a CPU.


## Task 2a: Data Download

**Aim:** Get tomograms with ribosome annotations for inference and evaluation.

We recommend to use `copick` and the `cryoet-data-portal` client to fetch the picks and the tomograms, respectively; see TopCUP's [quickstart notebook](https://github.com/czimaginginstitute/czii_cryoet_mlchallenge_winning_models/blob/main/notebooks/Quickstart_TopCUP.ipynb) for examples. Please download the **specific runs listed below** from the CZ cryoET Data Portal (only download the data with the given voxel size). This should be approximately 3GB of data. **Use exactly these runs, do not substitute.**

| Role | Description | Deposition ID | Dataset ID | Run ID | Voxel size | Run page |
|------|-----------|---------|-----|-------|------|------|
| In-distribution sanity check | CZII synthetic, lysate | 10310 | 10440 | 16463 | 10.012 Å | [link](https://cryoetdataportal.czscience.com/runs/16463?deposition-id=10310) |
| Held-out evaluation | Chlamydomonas, in-situ | 10300 | 10301 | 14070 | 7.84 Å | [link](https://cryoetdataportal.czscience.com/runs/14070?deposition-id=10300) |

The CZCDP-10310 run is from the **synthetic training set** TopCUP was trained on (in-distribution sanity check). The CZCDP-10300 run is an in-situ Chlamydomonas tomogram with **ribosome** ground-truth point annotations.

In your README, confirm the run IDs you used and report tomogram dimensions, voxel size, and number of ribosome annotations for each.

**Hints**
- Fetch tomograms with `cryoet_data_portal.Tomogram.download_omezarr(dest_path=...)`.
- Fetch picks via copick: use `copick.from_czcdp_datasets(...)` and then `pick.load()` on a run's picks.
- Persisting both locally and pointing `topcup inference` at a `"config_type": "filesystem"` copick config will save you re-streaming the tomogram from S3 on every re-run.

## Task 2b: Inference and Quality Assessment on Synthetic Data

**Aim:** Run TopCUP and report a quality assessment on the in-distribution synthetic tomogram.

Set up TopCUP (see the [tutorial](https://virtualcellmodels.cziscience.com/tutorial/topcup-tutorial)) and download the pretrained **`topcup_phantom_24_tomograms.ckpt`** weights from [Hugging Face](https://huggingface.co/kevinzhao/TopCUP/tree/main/topcup_weights). Write a script that runs inference with these weights on a given dataset. **Note**: TopCUP is hard-coded to use an NVIDIA GPU but many people don't have access to one. Find a way to work around that limitation so that your script is runnable on both a CPU or GPU. If you have an M-series Mac, it can be tempting to try to also write an mps-enabled version. Feel free to do so, but be aware that this is not a goal of this exercise and will therefore not be evaluated.

Run inference on the CZCDP-10310 synthetic tomogram from Task 1a—this is in the model's training distribution—and address the following:

1. **Inference-time hyperparameters.** Choose and justify the hyperparameters that affect what gets counted as a correct detection (e.g., matching radius, confidence threshold, tiling and aggregation choices). One or two sentences per choice on what informed it.
2. **Quality assessment.** Report metrics that capture both *detection* quality and *localization* quality. Pick the specific metrics and the operating point(s) yourself, and include a short justification for each metric. Report your best set of hyperparameters.


## Task 2c: Held-out Inference and Failure Characterization

**Aim:** Run the same model on the in-situ Chlamydomonas tomogram, diagnose where and why it fails, and reason about how to fix it.

Running TopCUP at full resolution on the whole Chlamydomonas tomogram is likely infeasible on a single-GPU or CPU setup due to memory and time constraints. To keep your time budget on diagnosis rather than memory management, run inference on the following **chunk** of run 14070: voxel offset `(x=160, y=752)`, size `256` in both x and y, full z. This area contains a large number of the ground-truth ribosomes (56 of the 202).

Use this chunk as-is. Report the same metrics you chose in Task 1b on this chunk, alongside the synthetic-data numbers, so the two are directly comparable.
Where and why does the model fail on the in-situ Chlamydomonas tomogram? Use the model's outputs—predictions, logits, intermediate activations, other outputs of your choice—to support your answer. Quantitative breakdowns and/or figures are encouraged where they help.
What are the differences between the cellular setting and the lysate synthetic data that might contribute to the difference in model performance?


## Task 2d: Proposal: Improving Performance

**Aim:** Propose a pipeline to improve model performance on the Chlamydomonas tomogram.

As found in the previous task, TopCUP doesn't generalize well to in-situ cellular data. The Chlamydomonas tomogram comes from a real biological samples and was acquired on a real CryoET microscope. Propose how you would extend the TopCUP pipeline to robustly detect ribosomes in this setting. There is no clean large-scale public training set for this regime; existing pickers cannot be treated as unbiased ground truth.

Given the failure modes you found in the previous task, how would you adapt the model and/or the training pipeline to close the gap to in-situ cellular data? How would you evaluate performance improvement?

A detailed technical outline is sufficient. Keep practical considerations and constraints in mind. No implementation needed.
