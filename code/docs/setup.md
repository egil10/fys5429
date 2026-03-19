# Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verify

```bash
cd code/scripts
python bs.py        # prints call/put, saves bs_surface.pdf
python heston.py    # takes ~1 min for 40x40 surface
python generate.py  # regenerates all parquet files
```

## GPU (optional)

PyTorch will auto-detect CUDA. Force CPU:

```python
model = BSPINN(device="cpu")
```
