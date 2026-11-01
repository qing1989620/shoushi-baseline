import hashlib, json, re, zipfile
from pathlib import Path
from collections import Counter
from io import BytesIO
from PIL import Image
root = Path('data/raw')
archives = [('rps.zip','https://storage.googleapis.com/download.tensorflow.org/data/rps.zip','rps'), ('rps-test-set.zip','https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip','rps-test-set')]
report = {'description':'Reconstructed from TensorFlow-hosted source images; not the verified hand.tar archive. Resized to 128x128 using Pillow LANCZOS; RGB PNG.', 'archives': [], 'counts': {}}
for archive, url, split in archives:
    path=root/'downloads'/archive
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''): sha.update(chunk)
    report['archives'].append({'url':url,'bytes':path.stat().st_size,'sha256':sha.hexdigest()})
    counts=Counter()
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad: raise ValueError(f'Corrupt member: {bad}')
        for item in z.infolist():
            match=re.fullmatch(r'(rps|rps-test-set)/(rock|paper|scissors)/([\w-]+\.png)',item.filename)
            if not match or match.group(1)!=split: continue
            _, label, name=match.groups()
            dst=root/'hand'/split/label/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(BytesIO(z.read(item))) as img:
                img.convert('RGB').resize((128,128),Image.Resampling.LANCZOS).save(dst,format='PNG')
            counts[label]+=1
    report['counts'][split]=dict(counts)
    print(split,dict(counts),flush=True)
(root/'provenance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
