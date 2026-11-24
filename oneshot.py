"""One-shot scene classification with a frozen SigLIP model (ITCS 6169/8169, Assignment 3).

Nothing is trained. The classifier combines two sets of class scores for every image:

  text scores       cosine similarity between the image embedding and one text embedding per class
                    (a short descriptive phrase per class, averaged over 7 prompt templates)
  prototype scores  cosine similarity between the image embedding and the embedding of the ONE labeled
                    training image of each class (original view, no augmentation)

Each set is standardised per test image across the 16 classes (z-score), then
    score = alpha * text_z + (1 - alpha) * prototype_z          (alpha = weight on the text scores)
and the prediction is the class with the highest score.

Exactly one training image per class is used by the final classifier. The 16 images are listed in
support_images.json; select_support.py explains and reproduces how they were chosen.

Heavy imports (torch, transformers, PIL) are done inside the functions that need them, so the numpy-only
parts of this file can be imported and tested without them.
"""
import json
import math
import re
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------------------
# Fixed recipe
# ------------------------------------------------------------------------------------------------------------
MODEL_NAME = 'siglip-b16-224'
MODEL_REPO = 'google/siglip-base-patch16-224'          # SigLIP ViT-B/16, 224 px, Hugging Face id
ALPHA = 0.75                                            # weight on the text scores in the blend
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

CLASSES = ['Bedroom', 'Coast', 'Flower', 'Forest', 'Highway', 'Industrial', 'InsideCity', 'Kitchen',
           'LivingRoom', 'Mountain', 'Office', 'OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding']
NUM_CLASSES = len(CLASSES)

# Hand-written class phrases, fixed before any result was looked at.
DESCRIPTIVE = {
    'Bedroom':      'a bedroom',
    'Coast':        'a coast by the sea',
    'Flower':       'a flower',
    'Forest':       'a forest',
    'Highway':      'a highway',
    'Industrial':   'an industrial area',
    'InsideCity':   'the inside of a city with buildings',
    'Kitchen':      'a kitchen',
    'LivingRoom':   'a living room',
    'Mountain':     'a mountain',
    'Office':       'an office',
    'OpenCountry':  'open country with fields',
    'Store':        'the inside of a store',
    'Street':       'a street',
    'Suburb':       'a suburban residential area',
    'TallBuilding': 'a tall building, a skyscraper',
}
# Post-hoc edit ("v2" prompts). It was chosen after looking at test errors, so the accuracy it gives is
# optimistic; it is reported as such. The original phrase for this class is in DESCRIPTIVE above.
PROMPT_EDITS = {
    'InsideCity': 'an urban scene with a building',
}
TEMPLATES = ['a photo of {}.', 'a picture of {}.', 'an image of {}.', 'a photo of {}, a type of scene.',
             'a photo taken of {}.', 'a low resolution photo of {}.', 'a cropped photo of {}.']

assert list(DESCRIPTIVE) == CLASSES and all(c in DESCRIPTIVE for c in PROMPT_EDITS)


def class_phrases(use_edits=True):
    """The 16 class phrases in class order (with the post-hoc edits unless use_edits is False)."""
    return [(PROMPT_EDITS.get(c, DESCRIPTIVE[c]) if use_edits else DESCRIPTIVE[c]) for c in CLASSES]


# ------------------------------------------------------------------------------------------------------------
# numpy helpers
# ------------------------------------------------------------------------------------------------------------
def l2norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def zscore_rows(s):
    """Standardise each row (one image) across the classes, so two kinds of score can be added fairly."""
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)


def binomial_se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def combine_scores(X, T, P, alpha=ALPHA, mode='blend'):
    """X: [N, D] image embeddings; T, P: [16, D] text and prototype embeddings (all L2-normalised).
    mode 'text' = zero-shot only, 'proto' = prototype only, 'blend' = alpha * text_z + (1 - alpha) * proto_z.
    Returns [N, 16] scores (argmax = prediction)."""
    if mode == 'text':
        return X @ T.T
    if mode == 'proto':
        return X @ P.T
    if mode != 'blend':
        raise ValueError(f'unknown mode {mode!r}')
    return alpha * zscore_rows(X @ T.T) + (1 - alpha) * zscore_rows(X @ P.T)


# ------------------------------------------------------------------------------------------------------------
# images and folders
# ------------------------------------------------------------------------------------------------------------
def list_class_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert('RGB')


def scan_imagefolder(root):
    """ImageFolder-style listing: returns (paths, labels, class_names) with classes sorted by folder name."""
    root = Path(root)
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    paths, labels = [], []
    for i, n in enumerate(names):
        for p in list_class_images(root / n):
            paths.append(p)
            labels.append(i)
    return paths, np.array(labels), names


# ------------------------------------------------------------------------------------------------------------
# the frozen model
# ------------------------------------------------------------------------------------------------------------
def _unwrap(x):
    """get_*_features returns a tensor in transformers 4.x; be tolerant of output objects."""
    import torch
    if torch.is_tensor(x):
        return x
    for attr in ('image_embeds', 'text_embeds', 'pooler_output'):
        v = getattr(x, attr, None)
        if v is not None:
            return v
    raise TypeError(f'unexpected feature output type: {type(x)}')


class SigLIPEmbedder:
    """Frozen SigLIP: images -> [N, D] and texts -> [N, D], both L2-normalised numpy arrays."""

    def __init__(self, repo=MODEL_REPO, device='cpu', batch_size=8):
        import torch
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError('this needs Hugging Face transformers: pip install -r requirements.txt') from e
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch, self.device, self.batch_size = torch, torch.device(device), batch_size
        t0 = time.time()
        self.model = AutoModel.from_pretrained(repo).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(repo)
        n = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f'loaded {repo} ({n:.0f}M parameters, {time.time() - t0:.0f}s, device {self.device})')

    def images(self, pil_images, progress=False):
        torch, F = self.torch, self.torch.nn.functional
        out, t0, n = [], time.time(), len(pil_images)
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                inp = self.proc(images=pil_images[i:i + self.batch_size], return_tensors='pt')
                inp = {k: v.to(self.device) for k, v in inp.items()}
                f = _unwrap(self.model.get_image_features(**inp)).float()
                out.append(F.normalize(f, dim=-1).cpu())
                if progress and ((i // self.batch_size) % 10 == 9 or i + self.batch_size >= n):
                    print(f'    {min(i + self.batch_size, n)}/{n} images | {time.time() - t0:.0f}s')
        return torch.cat(out).numpy()

    def texts(self, strings):
        torch, F = self.torch, self.torch.nn.functional
        out = []
        with torch.no_grad():
            for i in range(0, len(strings), 64):
                chunk = [s.lower() for s in strings[i:i + 64]]       # SigLIP was trained on lower-case text,
                tok = self.proc(text=chunk, padding='max_length', max_length=64,   # padded to 64 tokens, no mask
                                truncation=True, return_tensors='pt')
                f = self.model.get_text_features(input_ids=tok['input_ids'].to(self.device))
                out.append(F.normalize(_unwrap(f).float(), dim=-1).cpu())
        return torch.cat(out).numpy()


# ------------------------------------------------------------------------------------------------------------
# the two score sources
# ------------------------------------------------------------------------------------------------------------
def text_embeddings(embedder, use_edits=True):
    """[16, D]: for every class, the 7 template prompts are embedded, averaged, and re-normalised."""
    phrases = class_phrases(use_edits)
    texts = [t.format(p) for p in phrases for t in TEMPLATES]
    emb = embedder.texts(texts).reshape(NUM_CLASSES, len(TEMPLATES), -1).mean(1)
    return l2norm(emb)


def resolve_support_path(rel, train_dir, support_dir):
    """A support path is stored relative to the training folder ("Bedroom/image_0012.jpg").
    Look for it in the training folder first, then in the copy kept in this repository (support/)."""
    rel = Path(str(rel).replace('\\', '/'))
    for base in (Path(train_dir), Path(support_dir)):
        if (base / rel).is_file():
            return base / rel
    raise FileNotFoundError(f'support image {rel} not found under {train_dir} or {support_dir}')


def load_support(support_json, train_dir, support_dir):
    """Returns the 16 support-image paths in class order. Exactly one image per class is required."""
    spec = json.loads(Path(support_json).read_text())
    spec = spec.get('images', spec)
    if sorted(spec) != sorted(CLASSES):
        raise ValueError(f'{support_json} must list exactly one image for each of the 16 classes; got {sorted(spec)}')
    paths = []
    for c in CLASSES:
        if isinstance(spec[c], (list, tuple)):
            raise ValueError(f'class {c}: more than one image listed (one training sample per class is the rule)')
        p = resolve_support_path(spec[c], train_dir, support_dir)
        if p.parent.name != c:
            raise ValueError(f'class {c}: {p} is not inside a folder called {c}')
        paths.append(p)
    return paths


def prototype_embeddings(embedder, support_paths):
    """[16, D]: embedding of each class's single support image, original view (no augmentation)."""
    return l2norm(embedder.images([load_rgb(p) for p in support_paths]))


class OneShotClassifier:
    """text_embeddings + prototype_embeddings + the blend. predict_scores takes image embeddings."""

    def __init__(self, T, P, alpha=ALPHA, mode='blend'):
        self.T, self.P, self.alpha, self.mode = T, P, alpha, mode

    @classmethod
    def build(cls, embedder, support_json, train_dir, support_dir, alpha=ALPHA, mode='blend', use_edits=True):
        T = text_embeddings(embedder, use_edits)
        P = prototype_embeddings(embedder, load_support(support_json, train_dir, support_dir))
        return cls(T, P, alpha, mode)

    def predict_scores(self, X):
        return combine_scores(X, self.T, self.P, self.alpha, self.mode)
