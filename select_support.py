#!/usr/bin/env python3
"""Choose (or record) the one training image per class used by the final classifier (strategy S1, "typical").

The rule is label-free and text-free. For each class:
  1. a candidate pool of POOL_PER_CLASS training images is drawn at random with a seeded generator
     (numpy default_rng(POOL_SEED); the sorted file listing of the class folder is the population);
  2. every candidate is embedded with the frozen SigLIP image encoder (original view);
  3. the candidate with the highest MEAN COSINE SIMILARITY to the other candidates of the same class is kept
     (the most typical-looking image of the class).
Only these 16 images are given to the final classifier; the rest of the pool is discarded after the choice.
No test image and no test label is used here.

NOTE: building a pool means looking at POOL_PER_CLASS candidates per class to pick one. The assignment forbids using more
than one training sample per class in the classifier; the final classifier does use exactly one, but the pool step is
disclosed in the report.

Usage
-----
    # 1) re-derive the choice from scratch (needs the model; a few minutes on CPU) and write support_images.json
    python select_support.py --copy

    # 2) same, and check it against the file written by the notebook (the "S1" entry of selected_images.json)
    python select_support.py --check-against a3_results/selected_images.json

    # 3) no model needed: convert the notebook's selected_images.json (S1 entry) into support_images.json
    python select_support.py --from-notebook a3_results/selected_images.json --copy

--copy also copies the 16 images to support/<Class>/<file>, so the repository is self-contained.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import oneshot as O

POOL_PER_CLASS = 20
POOL_SEED = 5000


def relative_name(p, cls):
    """'data\\train\\Bedroom\\img.jpg' or 'data/train/Bedroom/img.jpg' -> 'Bedroom/img.jpg'."""
    parts = str(p).replace('\\', '/').split('/')
    if len(parts) < 2 or parts[-2] != cls:
        raise ValueError(f'{p}: expected the file to sit in a folder called {cls}')
    return f'{cls}/{parts[-1]}'


def draw_pool(train_dir):
    """Returns pool[c] = list of K image paths, exactly as the selection notebook drew them."""
    rng = np.random.default_rng(POOL_SEED)
    pool = []
    for c in O.CLASSES:
        files = O.list_class_images(Path(train_dir) / c)
        pick = np.sort(rng.choice(len(files), size=POOL_PER_CLASS, replace=False))
        pool.append([files[i] for i in pick])
    return pool


def select_typical(E):
    """E: [classes, K, D] L2-normalised candidate embeddings -> index of the most typical candidate per class."""
    K = E.shape[1]
    idx = []
    for c in range(E.shape[0]):
        G = E[c] @ E[c].T
        typ = (G.sum(1) - np.diag(G)) / (K - 1)                  # mean cosine to the other candidates of the class
        idx.append(int(typ.argmax()))
    return np.array(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-dir', default='data/train')
    ap.add_argument('--out', default='support_images.json')
    ap.add_argument('--support-dir', default='support', help='where --copy puts the 16 images')
    ap.add_argument('--copy', action='store_true', help='copy the selected images into --support-dir/<Class>/')
    ap.add_argument('--from-notebook', metavar='JSON', help="convert the notebook's selected_images.json instead of re-deriving")
    ap.add_argument('--strategy', default='S1', help='entry of the notebook file to convert (S1 is the final recipe)')
    ap.add_argument('--check-against', metavar='JSON', help="compare the re-derived choice with the notebook's selected_images.json")
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = ap.parse_args(argv)

    if args.from_notebook:
        sel = json.loads(Path(args.from_notebook).read_text())[args.strategy]
        chosen = {c: relative_name(sel[c], c) for c in O.CLASSES}
        source = f'converted from {args.from_notebook} [{args.strategy}]'
    else:
        pool = draw_pool(args.train_dir)
        embedder = O.SigLIPEmbedder(O.MODEL_REPO, args.device, args.batch_size)
        print(f'embedding the pool: {O.NUM_CLASSES} classes x {POOL_PER_CLASS} candidates ...')
        flat = [O.load_rgb(p) for row in pool for p in row]
        E = O.l2norm(embedder.images(flat, progress=True)).reshape(O.NUM_CLASSES, POOL_PER_CLASS, -1)
        idx = select_typical(E)
        chosen = {c: relative_name(pool[ci][int(idx[ci])], c) for ci, c in enumerate(O.CLASSES)}
        source = f're-derived: pool seed {POOL_SEED}, {POOL_PER_CLASS} candidates per class, strategy S1 (typical)'
        if args.check_against:
            ref = json.loads(Path(args.check_against).read_text())['S1']
            ref = {c: relative_name(ref[c], c) for c in O.CLASSES}
            diff = [c for c in O.CLASSES if ref[c] != chosen[c]]
            print('check against the notebook: ' + ('all 16 images agree' if not diff else
                  f'{len(diff)} differ: ' + ', '.join(f'{c} (notebook {ref[c]}, here {chosen[c]})' for c in diff)))
            if diff:
                print('A difference means two candidates were nearly tied and floating-point noise between runs broke the tie; '
                      'the notebook\'s choice is the one the reported numbers use. Use --from-notebook to keep it.')

    out = dict(images=chosen, model=O.MODEL_REPO, strategy='S1 (typical: highest mean cosine to the other candidates of the class)',
               pool_per_class=POOL_PER_CLASS, pool_seed=POOL_SEED, source=source,
               note='paths are relative to the training folder (data/train) or to support/')
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f'wrote {args.out}')
    for c in O.CLASSES:
        print(f'  {c:<14s} {chosen[c]}')

    if args.copy:
        n = 0
        for c, rel in chosen.items():
            src = Path(args.train_dir) / rel
            if not src.is_file():
                sys.exit(f'cannot copy {src}: not found (run from the project root, or pass --train-dir)')
            dst = Path(args.support_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
        print(f'copied {n} images to {args.support_dir}/')


if __name__ == '__main__':
    main()
