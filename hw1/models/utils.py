from pathlib import Path
import gzip
import hashlib
import html
import json
import re
import subprocess
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/'
SOURCES = {
    'Video_Games.csv': BASE + 'categoryFilesSmall/Video_Games.csv',
    'meta_Video_Games.json.gz': BASE + 'metaFiles2/meta_Video_Games.json.gz',
}
SEED = 42
CATALOG_SIZE = 4000
MAX_USERS = 40000


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    expected = {
        'Video_Games.csv': 'c3fec1cce4ee67f0d51f8b53a7e294d8bbe6f75ca1d8732256e900b3c46606d8',
        'meta_Video_Games.json.gz': '4069e17c7efa33e7574c2e77f1fe0b0c6f2a42a7ac29c4422696d788d8a8e0a6',
    }
    manifest = {}
    for name, url in SOURCES.items():
        target = DATA / name
        if not target.exists():
            part = target.with_suffix(target.suffix + '.part')
            subprocess.run(['curl', '-fL', '--retry', '3', '--connect-timeout', '20',
                            '--max-time', '600', url, '-o', str(part)], check=True)
            part.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if name in expected and digest != expected[name]:
            raise ValueError(f'Checksum mismatch for {name}; remove the incomplete local file and retry')
        manifest[name] = dict(url=url, bytes=target.stat().st_size, sha256=digest)
    return manifest


def prepare():
    raw = pd.read_csv(DATA / 'Video_Games.csv', header=None,
        names=['item_id', 'user_id', 'rating', 'timestamp'],
        dtype={'item_id': str, 'user_id': str})
    report = {'raw_rows': len(raw), 'raw_users': raw.user_id.nunique(),
              'raw_items': raw.item_id.nunique(),
              'missing': raw.isna().sum().to_dict(),
              'exact_duplicates': int(raw.duplicated().sum())}
    clean = raw.dropna().copy()
    clean = clean[(clean.rating.between(1, 5)) & (clean.timestamp > 0)]
    clean = clean.sort_values(['timestamp', 'user_id', 'item_id'], kind='stable')
    report['repeated_user_item_rows'] = int(clean.duplicated(['user_id', 'item_id']).sum())
    clean = clean.drop_duplicates(['user_id', 'item_id'], keep='first')
    t0, t1 = np.quantile(clean.timestamp, [.70, .85]).astype(np.int64)
    early = clean[clean.timestamp < t0]
    counts = early.groupby('item_id').size().rename('count').reset_index()
    catalog = counts.sort_values(['count', 'item_id'], ascending=[False, True]).head(CATALOG_SIZE).item_id
    selected = clean[clean.item_id.isin(catalog)].copy()
    early = selected[(selected.timestamp < t0) & (selected.rating >= 4)]
    counts = early.groupby('user_id').size()
    users = sorted(counts[counts >= 2].index)
    users = sorted(sorted(users, key=lambda u: hashlib.sha256(
        f'{SEED}:{u}'.encode()).hexdigest())[:MAX_USERS])
    selected = selected[selected.user_id.isin(users)].copy()
    items = sorted(selected.loc[selected.timestamp < t0, 'item_id'].unique())
    selected = selected[selected.item_id.isin(items)]
    umap, imap = {u:i for i,u in enumerate(users)}, {v:i for i,v in enumerate(items)}
    selected['user_idx'] = selected.user_id.map(umap).astype(int)
    selected['item_idx'] = selected.item_id.map(imap).astype(int)
    train = selected[selected.timestamp < t0].reset_index(drop=True)
    val = selected[(selected.timestamp >= t0) & (selected.timestamp < t1)].reset_index(drop=True)
    test = selected[selected.timestamp >= t1].reset_index(drop=True)
    shape = (len(users), len(items))
    assert 100_000 <= len(selected) <= 2_000_000, len(selected)
    assert min(shape) >= 1000, shape
    assert train.timestamp.max() < val.timestamp.min() <= val.timestamp.max() < test.timestamp.min()
    assert not selected.duplicated(['user_idx', 'item_idx']).any()
    report.update(clean_rows=len(clean), selected_rows=len(selected),
        users=shape[0], items=shape[1], sparsity=1 - len(selected) / np.prod(shape),
        train_rows=len(train), validation_rows=len(val), test_rows=len(test),
        train_end_exclusive=int(t0), validation_end_exclusive=int(t1),
        excluded_rows_after_cleaning=len(clean) - len(selected))
    metadata = {}
    meta_rows = 0
    with gzip.open(DATA / 'meta_Video_Games.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            meta_rows += 1
            asin = d.get('asin')
            if asin in imap and asin not in metadata:
                metadata[asin] = {k:d.get(k, '') for k in ['title', 'brand', 'category']}
    meta = pd.DataFrame([dict(item_id=i, **metadata.get(i, {})) for i in items])
    for col in ['title', 'brand', 'category']:
        if col not in meta:
            meta[col] = ''
    report['metadata_source_rows'] = meta_rows
    report['metadata_item_coverage'] = len(metadata) / len(items)
    report['brand_coverage'] = float(meta.brand.fillna('').astype(bool).mean())
    report['category_coverage'] = float(meta.category.fillna('').astype(bool).mean())
    features, feature_report = encode_metadata(meta)
    report['features'] = feature_report
    return dict(raw=raw, events=selected, train=train, validation=val, test=test,
                metadata=meta, item_features=features, shape=shape,
                t0=int(t0), t1=int(t1), report=report)


def normalize(value):
    if not isinstance(value, str):
        return ''
    return re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]+>', ' ', value))).strip().lower()


def encode_metadata(meta):
    n = len(meta)
    brands = meta.brand.map(normalize)
    counts = brands.value_counts()
    common = sorted(v for v in counts[counts >= 2].index if v)
    bmap = {s:i + 1 for i,s in enumerate(common)}
    tokens = [set(re.findall(r'[a-z]{3,}', normalize(s))) for s in meta.title]
    freq = Counter(t for words in tokens for t in words)
    stop = {'the','and','with','from','feat','featuring','album','version',
            'span','class','size','normal','color','secondary','text','parentaladvisory'}
    vocab = sorted((t for t,c in freq.items() if c >= 5 and t not in stop),
                   key=lambda t:(-freq[t], t))[:2048]
    tmap = {s:i + 1 for i,s in enumerate(vocab)}
    keyword = [min((t for t in words if t in tmap), key=lambda t:(freq[t], t),
                   default='') for words in tokens]
    brand = np.array([n + bmap.get(s, 0) for s in brands])
    title = np.array([n + len(bmap) + 1 + tmap.get(s, 0) for s in keyword])
    categories = [normalize(v[-1]) if isinstance(v, list) and v else '' for v in meta.category]
    cmap = {s:i + 1 for i,s in enumerate(sorted(set(categories) - {''}))}
    category = np.array([n + len(bmap) + len(tmap) + 2 + cmap.get(s, 0) for s in categories])
    return np.column_stack([brand, title, category]), dict(
        brand_values=len(bmap), title_values=len(tmap), category_values=len(cmap),
        brand_encoded_coverage=float(np.mean([s in bmap for s in brands])),
        title_encoded_coverage=float(np.mean([bool(s) for s in keyword])))


def matrix(events, shape, positive_only=False):
    d = events[events.rating >= 4] if positive_only else events
    return sparse.csr_matrix((np.ones(len(d)), (d.user_idx, d.item_idx)), shape=shape)


def evaluation_context(past, heldout, shape):
    seen = matrix(past, shape)
    warm_items = np.asarray(seen.sum(axis=0)).ravel() > 0
    warm_users = np.asarray(seen.sum(axis=1)).ravel() > 0
    relevant = heldout[(heldout.rating >= 4)].copy()
    eligible = warm_users[relevant.user_idx] & warm_items[relevant.item_idx]
    warm = relevant[eligible]
    unseen = np.array([seen[u, i] == 0 for u,i in zip(warm.user_idx, warm.item_idx)])
    warm = warm[unseen]
    truth = {int(u):g.item_idx.to_numpy() for u,g in warm.groupby('user_idx')}
    users = np.array(sorted(truth), dtype=int)
    report = dict(relevant_rows=len(relevant), eligible_relevant_rows=len(warm),
        excluded_cold_rows=int((~eligible).sum()), excluded_seen_rows=int((~unseen).sum()),
        evaluated_users=len(users), candidate_catalog=int(warm_items.sum()))
    assert len(users), report
    return dict(seen=seen, warm_items=warm_items, truth=truth, users=users, report=report)


def recommend(model, users, context, k=10):
    scores = np.asarray(model.score(users), dtype=float)
    assert np.isfinite(scores).all(), 'Non-finite model scores'
    scores[:, ~context['warm_items']] = -np.inf
    past = context['seen'][users].tocoo()
    scores[past.row, past.col] = -np.inf
    assert (np.isfinite(scores).sum(axis=1) >= k).all()
    return np.argsort(-scores, axis=1, kind='stable')[:, :k]


def evaluate(model, context, k=10, batch_size=128):
    per_user = []
    for start in range(0, len(context['users']), batch_size):
        users = context['users'][start:start + batch_size]
        top = recommend(model, users, context, k)
        for u, recs in zip(users, top):
            truth = context['truth'][int(u)]
            hit = np.isin(recs, truth)
            dcg = float(np.sum(hit / np.log2(np.arange(2, k + 2))))
            idcg = float(np.sum(1 / np.log2(np.arange(2, min(k, len(truth)) + 2))))
            per_user.append((int(u), dcg / idcg, hit.sum() / len(truth)))
    rows = pd.DataFrame(per_user, columns=['user_idx', 'NDCG@10', 'Recall@10'])
    metrics = rows.drop(columns='user_idx').mean().to_dict()
    return metrics, rows
