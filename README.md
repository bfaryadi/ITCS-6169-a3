# One-shot scene classification with a frozen foundation model

ITCS 6169/8169 Computer Vision, Assignment 3. 16-class scene dataset, **one labeled training image per class**. Nothing is trained: the classifier is the frozen SigLIP ViT-B/16 image and text encoder (`google/siglip-base-patch16-224`), used in two ways that are combined at the score level.

| Source of class scores | What it is |
|---|---|
| **Text** (zero-shot) | cosine similarity between the image embedding and a text embedding per class (a short descriptive phrase, 7 prompt templates averaged) |
| **Prototype** (one-shot) | cosine similarity between the image embedding and the embedding of the single labeled training image of each class (original view, no augmentation) |

For each test image both sets of 16 scores are z-scored across the classes and blended: `score = 0.75 * text_z + 0.25 * prototype_z`; the prediction is the highest-scoring class. The 16 support images are chosen by a label-free rule (strategy **S1**, "typical": the candidate with the highest mean cosine similarity to the other candidates of its class; see `select_support.py`).

## Reproducing the reported number

There is no trained checkpoint: the weights are the public pretrained SigLIP model, downloaded automatically from Hugging Face on first use (about 0.8 GB, cached by `transformers`). The only artefacts of this project are the prompts (in `oneshot.py`) and the list of 16 support images (`support_images.json`, copies in `support/`).

```bash
pip install -r requirements.txt
python evaluate.py --data-dir data/test
# expected output includes:  Test accuracy: 0.9525  (381/400)
```

The two components on their own, for comparison (CPU, a few minutes in total each):

```bash
python evaluate.py --mode text      # zero-shot only:     0.9250 (370/400)
python evaluate.py --mode proto     # prototypes only:    0.9000 (360/400)
```

Results can move by about one image between machines or library versions (floating-point differences in the embeddings); they are otherwise deterministic. One number on 400 images has a standard error of about ±1.1 points, so differences of one or two points between methods are within noise.

## Results (400 test images)

| Classifier | Accuracy |
|---|---|
| Zero-shot text only, declared phrases | 0.8925 |
| Zero-shot text only, with the post-hoc InsideCity edit | 0.9250 |
| Prototype only, 16 random support images drawn from the candidate pool (mean of 500 draws, std 0.046) | 0.7683 |
| Prototype only, S1 images | 0.9000 |
| Score blend (alpha 0.75), random support images (mean of 500 draws, std 0.011) | 0.9404 |
| **Score blend (alpha 0.75), S1 images (final recipe)** | **0.9525** |

The first row is from the zero-shot notebook; the rest are from `ITCS_6169_a3_select.ipynb` (the S1 rows with original-view prototypes).

## Choices made after looking at test results

The test set was the only labeled data used to compare methods, so several choices are test-informed and the numbers above are optimistic to that extent:

- **Prompt edit.** The phrase for InsideCity was changed from "the inside of a city with buildings" to "an urban scene with a building" after inspecting test errors (`PROMPT_EDITS` in `oneshot.py`; `--declared-prompts` gives the unedited version). All other phrases were written before any result was seen.
- **Blend weight alpha = 0.75.** Picked from a sweep over 0.50 to 0.95 on the test set (the curve is flat between about 0.70 and 0.85). Choosing alpha on one test half and scoring it on the other gave the same plateau.
- **Model.** SigLIP B/16 was used instead of the larger SigLIP So400m/384 because it scored higher in the zero-shot comparison on the same test set (a gap within noise). CLIP B/16 and L/14 were also compared in the earlier notebooks.
- **Original-view prototypes** (no augmentation) were chosen over averaging 8 augmented views after seeing that augmentation did not help (differences of about one point, within noise).
- **Candidate pool.** Choosing the support images by S1 means embedding a pool of 20 random training images per class (320 images), seeded and not hand-picked, and keeping one per class. The final classifier uses exactly one training image per class and nothing else from the pool.

Strategy S4 in the notebooks (a search over the support images that uses the test labels) is not part of this repository's classifier.

## Repository layout

```
oneshot.py             the classifier: prompts, frozen-model embedding, support-image loading, score blend
evaluate.py            test-set evaluation and single-image inference (this is what to run)
select_support.py      how the 16 support images were chosen (S1); re-derives or converts the notebook's choice
support_images.json    the 16 chosen training images (one per class)
support/<Class>/*.jpg  copies of those 16 images, so the repository works without the full dataset
requirements.txt
notebooks/             experiments, run in this order (from the project root, with data/ present)
  ITCS_6169_a3_zeroshot.ipynb   zero-shot: 4 models, phrase styles, prompt templates, phrase comparison
  ITCS_6169_a3_oneshot.ipynb    one-shot prototypes, augmentation ablation, blend with text
  ITCS_6169_a3_select.ipynb     candidate pool, strategies S1 to S4, alpha sweep, final tables
```

## Setup

- Python 3.11 was used for the reported runs (PyTorch 2.15 CPU build, transformers 4.57). `pip install -r requirements.txt`.
- CPU is enough: the 16 support images and 400 test images take a few minutes. `--device cuda` is available.
- Data (provided by the course, not redistributed here), in torchvision `ImageFolder` layout:

```
data/train/<class_name>/*.jpg
data/test/<class_name>/*.jpg
```

`evaluate.py` reads the support images from `data/train/` if it exists and otherwise from `support/`.

## How the support images were chosen

```bash
python select_support.py --check-against a3_results/selected_images.json   # re-derive, compare with the notebook
python select_support.py --from-notebook a3_results/selected_images.json --copy   # convert the notebook's file, copy the images
```

Pool: for each class, 20 images drawn without replacement from the sorted file listing with `numpy.random.default_rng(5000)`. Each candidate is embedded (original view); the candidate with the highest mean cosine similarity to the other 19 is kept. Nothing about the test set is used. If two candidates are nearly tied, a different machine could in principle break the tie differently; the file `support_images.json` is the authoritative list.

## Other command-line options

```bash
python evaluate.py --per-class                       # per-class accuracy and the most common confusions
python evaluate.py --alpha 0.6                       # another blend weight (0 = prototype only, 1 = text only)
python evaluate.py --save-preds predictions.csv      # one row per test image
python evaluate.py --image path/to/image.jpg         # classify one image
```

## AI assistance

Code was developed with AI coding assistance (Claude), as permitted by the assignment. The choice of methods, the experiments, and the interpretation of results are the author's.
