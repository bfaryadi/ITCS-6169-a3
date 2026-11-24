#!/usr/bin/env python3
"""Evaluation / inference for ITCS 6169/8169 Assignment 3 (one-shot scene classification, frozen SigLIP).

Builds the classifier (text embeddings of the class prompts + the embeddings of the 16 support images), then reports
accuracy on the test folder (default) or classifies a single image.

Examples
--------
    # the reported result: score blend, alpha 0.75, 16 selected support images
    python evaluate.py --data-dir data/test

    # the two components on their own (zero-shot text only; prototype only)
    python evaluate.py --mode text
    python evaluate.py --mode proto

    # a different blend weight, per-class accuracy, and the predictions saved to a CSV
    python evaluate.py --alpha 0.6 --per-class --save-preds predictions.csv

    # classify one image
    python evaluate.py --image some.jpg

Expected folder layout (torchvision ImageFolder): <data-dir>/<class_name>/<image files>.
Support images: support_images.json lists one training image per class, as paths relative to --train-dir
(or to --support-dir, the copy kept in this repository).
"""
import argparse
import csv
import sys

import numpy as np

import oneshot as O


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='data/test', help='ImageFolder with the test images')
    ap.add_argument('--train-dir', default='data/train', help='ImageFolder with the training images (support images are read from here)')
    ap.add_argument('--support-json', default='support_images.json', help='one training image per class (see select_support.py)')
    ap.add_argument('--support-dir', default='support', help='repository copy of the 16 support images, same Class/file layout (fallback)')
    ap.add_argument('--mode', choices=['blend', 'text', 'proto'], default='blend',
                    help="'blend' = reported classifier; 'text' = zero-shot only; 'proto' = support-image prototypes only")
    ap.add_argument('--alpha', type=float, default=O.ALPHA, help='weight on the text scores in the blend (0 to 1)')
    ap.add_argument('--declared-prompts', action='store_true',
                    help='use the originally declared phrases without the post-hoc edit (see oneshot.PROMPT_EDITS)')
    ap.add_argument('--image', help='classify a single image instead of evaluating the test folder')
    ap.add_argument('--topk', type=int, default=3, help='number of predictions to print for --image')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    ap.add_argument('--per-class', action='store_true', help='also print per-class accuracy and the most common confusions')
    ap.add_argument('--save-preds', metavar='CSV', help='write one row per test image (path, true class, predicted class)')
    args = ap.parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        sys.exit('--alpha must be between 0 and 1')

    embedder = O.SigLIPEmbedder(O.MODEL_REPO, args.device, args.batch_size)
    clf = O.OneShotClassifier.build(embedder, args.support_json, args.train_dir, args.support_dir,
                                    alpha=args.alpha, mode=args.mode, use_edits=not args.declared_prompts)

    print(f'Model      : {O.MODEL_NAME} ({O.MODEL_REPO}), frozen, no training')
    print(f'Classifier : {args.mode}' + (f' (alpha = {args.alpha:g} on the text scores)' if args.mode == 'blend' else ''))
    print(f'Prompts    : {"declared phrases" if args.declared_prompts else "declared phrases + post-hoc edits " + str(O.PROMPT_EDITS)}')
    print('Support    : one training image per class')
    for c, p in zip(O.CLASSES, O.load_support(args.support_json, args.train_dir, args.support_dir)):
        print(f'  {c:<14s} {p}')

    # ---- single image ----
    if args.image:
        x = embedder.images([O.load_rgb(args.image)])
        s = clf.predict_scores(x)[0]
        z = (s - s.max())
        prob = np.exp(z) / np.exp(z).sum()                       # softmax of the scores, a rough confidence only
        order = np.argsort(-s)[:min(args.topk, O.NUM_CLASSES)]
        print(f'\nPredictions for {args.image}:')
        for i in order:
            print(f'  {O.CLASSES[i]:<24s} score {s[i]:+.3f}   softmax {prob[i]:.3f}')
        return

    # ---- test folder ----
    paths, y, names = O.scan_imagefolder(args.data_dir)
    if names != O.CLASSES:
        sys.exit(f'test folders {names} do not match the expected classes {O.CLASSES}')
    print(f'\nembedding {len(paths)} test images ...')
    X = embedder.images([O.load_rgb(p) for p in paths], progress=True)
    pred = clf.predict_scores(X).argmax(1)

    correct = int((pred == y).sum())
    acc = correct / len(y)
    print(f'\nTest accuracy: {acc:.4f}  ({correct}/{len(y)})   standard error about ±{O.binomial_se(acc, len(y)):.4f}')

    if args.per_class:
        print('\nPer-class accuracy:')
        for i, n in enumerate(names):
            m = y == i
            print(f'  {n:<24s} {int((pred[m] == i).sum()):>3d}/{int(m.sum()):<3d} {(pred[m] == i).mean():.3f}')
        cm = np.zeros((O.NUM_CLASSES, O.NUM_CLASSES), dtype=int)
        np.add.at(cm, (y, pred), 1)                              # rows = true class, columns = predicted class
        off = sorted(((int(cm[i, j]), names[i], names[j]) for i in range(O.NUM_CLASSES) for j in range(O.NUM_CLASSES)
                      if i != j and cm[i, j] > 0), reverse=True)[:6]
        print('\nMost common confusions (true -> predicted):')
        for n, a, b in off:
            print(f'  {a} -> {b}: {n}')

    if args.save_preds:
        with open(args.save_preds, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['path', 'true', 'predicted', 'correct'])
            for p, t, q in zip(paths, y, pred):
                w.writerow([str(p), names[t], names[q], int(t == q)])
        print(f'\npredictions written to {args.save_preds}')


if __name__ == '__main__':
    main()
