import os
import base64
from pathlib import Path

import pytest
import requests


EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


def collect_images(root: Path, n: int):
    imgs = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() in EXTS:
                imgs.append(os.path.join(dirpath, fn))
                if len(imgs) >= n:
                    return imgs
    return imgs


def encode_image_b64(path: str) -> str:
    with open(path, 'rb') as f:
        data = f.read()
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def get_data_dirs():
    repo_root = Path(__file__).resolve().parents[1]
    pos_dir = repo_root / 'data' / 'Original'
    neg_dir = repo_root / 'data' / 'random_doc_images'
    return pos_dir, neg_dir


def server_url():
    return os.environ.get('TEST_SERVER_URL', 'http://127.0.0.1:8000')


def post_verify(b64_payload: str):
    url = server_url().rstrip('/') + '/verify'
    resp = requests.post(url, json={'image_base64': b64_payload})
    return resp


def test_positive_samples():
    n = int(os.environ.get('SAMPLE_N', '3'))
    pos_dir, _ = get_data_dirs()
    imgs = collect_images(str(pos_dir), n)
    if not imgs:
        pytest.skip(f"No positive images found in {pos_dir}")

    for p in imgs:
        b64 = encode_image_b64(p)
        resp = post_verify(b64)
        assert resp.status_code == 200
        j = resp.json()
        assert 'binary' in j
        assert 'predicted_label' in j['binary']
        assert 'probabilities' in j['binary']


def test_negative_samples():
    n = int(os.environ.get('SAMPLE_N', '3'))
    _, neg_dir = get_data_dirs()
    imgs = collect_images(str(neg_dir), n)
    if not imgs:
        pytest.skip(f"No negative images found in {neg_dir}")

    for p in imgs:
        b64 = encode_image_b64(p)
        resp = post_verify(b64)
        assert resp.status_code == 200
        j = resp.json()
        assert 'binary' in j
