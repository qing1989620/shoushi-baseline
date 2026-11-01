from pathlib import Path
import argparse, csv, hashlib, random, shutil, tarfile
import cv2
import numpy as np

CLASSES = {'rock': 0, 'scissors': 1, 'paper': 2}
EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def box_from_background(image):
    """Estimate a single hand box by contrast with the image border; review labels manually."""
    h, w = image.shape[:2]
    # Most source images have a light, nearly uniform background.
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)
    bg = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - bg, axis=2)
    mask = (distance > 30).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, bw, bh, area = stats[component]
    if area < h * w * 0.01:
        return None
    pad = max(2, round(min(h, w) * .02))
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(w, x+bw+pad), min(h, y+bh+pad)
    return (x1, y1, x2, y2), area / (h*w)


def main():
    p = argparse.ArgumentParser(description='Convert the official hand dataset to YOLO detection format (pseudo boxes).')
    p.add_argument('--source', required=True, help='Extracted hand dataset directory')
    p.add_argument('--output', default='data/rps')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if not source.is_dir(): p.error(f'Source directory does not exist: {source}')
    if output.exists() and any(output.iterdir()): p.error(f'Output is not empty: {output}. Choose a new output directory to preserve edits.')
    pools = {'trainval': {c: [] for c in CLASSES}, 'test': {c: [] for c in CLASSES}}
    for f in sorted(source.rglob('*')):
        if not f.is_file() or f.suffix.lower() not in EXTS or f.name.startswith('._'): continue
        parts = [x.lower() for x in f.relative_to(source).parts]
        labels = [x for x in parts[:-1] if x in CLASSES]
        if len(labels) != 1: continue
        if 'rps-test-set' in parts: split = 'test'
        elif 'rps' in parts: split = 'trainval'
        else: continue
        pools[split][labels[0]].append(f)
    if any(not pools[s][c] for s in pools for c in CLASSES):
        p.error('Expected rps/{rock,scissors,paper} and rps-test-set/{rock,scissors,paper} under source; other folders are not used.')
    # Detect exact duplicates across official training and test sets, rather than silently leaking.
    hashes = {}
    for split in pools:
        for cls, files in pools[split].items():
            for f in files:
                digest = hashlib.sha256(f.read_bytes()).hexdigest()
                if digest in hashes and hashes[digest][0] != split:
                    p.error(f'Identical image across source train/test: {hashes[digest][1]} and {f}')
                hashes[digest] = (split, f)
    randomizer = random.Random(args.seed)
    assignments = []
    for cls in CLASSES:
        files = pools['trainval'][cls][:]
        randomizer.shuffle(files)
        n_val = max(1, round(len(files) * .1))
        assignments.extend(('val', cls, f) for f in files[:n_val])
        assignments.extend(('train', cls, f) for f in files[n_val:])
        assignments.extend(('test', cls, f) for f in pools['test'][cls])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, cls, f in assignments:
        image = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f'Cannot read image: {f}')
        result = box_from_background(image)
        # Do not fabricate full-image boxes when segmentation fails.
        if result is None: raise ValueError(f'No foreground detected: {f}. Fix the source/segmentation before proceeding.')
        (x1,y1,x2,y2), foreground = result
        h,w = image.shape[:2]
        name = hashlib.sha256(str(f.relative_to(source)).encode('utf-8')).hexdigest()[:16]
        imdir, labeldir = output/'images'/split, output/'labels'/split
        imdir.mkdir(parents=True, exist_ok=True); labeldir.mkdir(parents=True, exist_ok=True)
        dest = imdir/(name+f.suffix.lower())
        shutil.copy2(f, dest)
        (labeldir/(name+'.txt')).write_text(f'{CLASSES[cls]} {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n', encoding='utf-8')
        rows.append((split, cls, str(f.relative_to(source)), str(dest.relative_to(output)), round(foreground,4), int(foreground > .85 or foreground < .03)))
    (output/'data.yaml').write_text(f"train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Rock\n  1: Scissors\n  2: Paper\n", encoding='utf-8')
    with (output/'manifest.csv').open('w', encoding='utf-8-sig', newline='') as out:
        writer=csv.writer(out); writer.writerow(['split','class','source','image','foreground_fraction','review_flag']); writer.writerows(rows)
    for s in ('train','val','test'): print(s, sum(row[0] == s for row in rows))
    print('Pseudo-labels generated. Review and correct all splits; flagged rows are not exhaustive:', output/'manifest.csv')

if __name__ == '__main__': main()
