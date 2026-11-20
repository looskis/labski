"""Submit eight reproducible, synthetic image-only inventory lookup tasks."""
import base64, hashlib, json, urllib.request
from pathlib import Path

def collect(base,model,destination):
    with Path(destination).open('x') as out:
        for seed in range(8):
            card=json.loads((Path(__file__).parent/'cards'/f'{seed}.json').read_bytes())
            image=base64.b64encode((Path(__file__).parent/'cards'/f'{seed}.png').read_bytes()).decode()
            prompt=f"Read the inventory shown in the image. Return only the six-digit number next to bin {card['target']}."
            content=[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+image}}]
            body=dict(model=model,messages=[dict(role='user',content=content)],temperature=0,max_tokens=64,chat_template_kwargs={'enable_thinking':False})
            req=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as reply:r=json.load(reply)
            c=r['choices'][0];text=c['message'].get('content') or ''
            record=dict(**card,finish=c['finish_reason'],correct=c['finish_reason']=='stop' and text.strip()==card['expected'],answer_sha256=hashlib.sha256(text.encode()).hexdigest(),usage=r['usage'])
            out.write(json.dumps(record,separators=(',',':'))+'\n');out.flush()


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('model')
    cli.add_argument('out', type=Path)
    cli.add_argument('--url', default='http://127.0.0.1:8000')
    args = cli.parse_args()
    collect(args.url, args.model, args.out)
