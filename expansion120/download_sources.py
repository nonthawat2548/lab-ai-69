"""Resumable public dataset download with validated HTTP ranges."""
import concurrent.futures, json, os, sys, time
from pathlib import Path
from urllib.request import Request,urlopen

def download(file_id,size,path,workers=8):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    marker=Path(str(path)+'.ranges.json');done=set(json.loads(marker.read_text())) if marker.exists() else set()
    if not path.exists():
        with path.open('wb') as f:f.truncate(size)
    if path.stat().st_size!=size:raise ValueError('Unexpected existing output size')
    chunk=64*1024*1024;total=(size+chunk-1)//chunk
    def one(i):
        start=i*chunk;end=min(size,start+chunk)-1
        req=Request(f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t',headers={'Range':f'bytes={start}-{end}'})
        for retry in range(3):
            try:
                with urlopen(req,timeout=90) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise IOError('Wrong range response')
                    position=start
                    while data:=r.read(1024*1024):
                        os.pwrite(fd,data,position);position+=len(data)
                    if position!=end+1:raise IOError('Incomplete chunk')
                return i
            except Exception:
                if retry==2:raise
    fd=os.open(path,os.O_RDWR)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(one,i) for i in range(total) if i not in done]):
                done.add(future.result());marker.write_text(json.dumps(sorted(done)))
                if len(done)%8==0 or len(done)==total: print(path.name,len(done),'/',total,flush=True)
    finally:os.close(fd)
    print('Complete',path,size,flush=True)
if __name__=='__main__':download(sys.argv[1],int(sys.argv[2]),sys.argv[3])
