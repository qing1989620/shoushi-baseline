from pathlib import Path
import argparse, random
import cv2
import numpy as np

p=argparse.ArgumentParser(description='Create a contact sheet for checking pseudo-boxes.')
p.add_argument('--data', default='data/rps')
p.add_argument('--split', choices=['train','val','test'], default='train')
p.add_argument('--count', type=int, default=60)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--output', default='label_review.jpg')
a=p.parse_args()
root=Path(a.data); files=sorted((root/'images'/a.split).glob('*'))
if not files: raise SystemExit('No images found; run prepare_dataset.py first.')
rng=random.Random(a.seed); files=rng.sample(files, min(a.count,len(files)))
size=160; cols=6; rows=(len(files)+cols-1)//cols
sheet=np.full((rows*(size+30),cols*size,3), 255, np.uint8)
for i, f in enumerate(files):
    im=cv2.imdecode(np.fromfile(str(f),dtype=np.uint8),cv2.IMREAD_COLOR)
    label=(root/'labels'/a.split/(f.stem+'.txt')).read_text().split()
    _,xc,yc,w,h=map(float,label[:5]); H,W=im.shape[:2]
    cv2.rectangle(im,(int((xc-w/2)*W),int((yc-h/2)*H)),(int((xc+w/2)*W),int((yc+h/2)*H)),(0,0,255),2)
    tile=cv2.resize(im,(size,size)); x=(i%cols)*size; y=(i//cols)*(size+30)
    sheet[y:y+size,x:x+size]=tile
    cv2.putText(sheet,f.stem[:10],(x+2,y+size+18),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1)
ok,encoded=cv2.imencode(Path(a.output).suffix or '.jpg',sheet)
if not ok: raise SystemExit('Cannot encode output image')
encoded.tofile(a.output)
print('Saved:',a.output,'; tile names are image filename prefixes. Inspect and correct labels.')
