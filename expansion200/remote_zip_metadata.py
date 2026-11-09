"""Read ZIP directory/label metadata over HTTP Range without downloading all images."""
import io,json,zipfile
from urllib.request import Request,urlopen
class RemoteZip(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.position=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('negative seek')
        return self.position
    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size-self.position)
        if n<=0:return b''
        start=self.position;end=start+n-1
        req=Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urlopen(req,timeout=45) as response:
            if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                raise IOError('Server did not return requested range')
            data=response.read(n)
        self.position+=len(data);return data

def inspect(file_id,size,prefix):
    from pathlib import Path
    from collections import Counter
    uri=f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
    with zipfile.ZipFile(RemoteZip(uri,size)) as z:
        infos=z.infolist()
        rows=[{'path':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size,'crc':x.CRC} for x in infos if not x.is_dir()]
        Path(prefix+'_directory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        texts=[x for x in infos if x.filename.lower().endswith(('.csv','.txt','.json','.md')) and x.file_size<100000]
        print('metadata files',[x.filename for x in texts],flush=True)
        for i,x in enumerate(texts):
            Path(prefix+f'_metadata_{i}.txt').write_bytes(z.read(x))
        image_folders=Counter('/'.join(x['path'].split('/')[:-1]) for x in rows if x['path'].lower().endswith(('.jpg','.jpeg','.png','.webp')))
        print(json.dumps({'files':len(rows),'images':sum(image_folders.values()),'image_folders':dict(image_folders)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    import sys
    inspect(sys.argv[1],int(sys.argv[2]),sys.argv[3])
