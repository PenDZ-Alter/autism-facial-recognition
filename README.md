# ASD Model Tester (local)

Flask web app for testing several `.keras` binary classifiers (0 = ASD, 1 = Non-ASD)
on one or many images. Runs only on `127.0.0.1`; images are processed in memory and never stored.

## Structure
```
asd-model-tester/
├── app.py
├── requirements.txt
├── models/              <- put your .keras files here
│   └── models.json      <- optional per-model settings (see models/models.json.example)
├── templates/index.html
└── static/
    ├── style.css
    └── app.js
```

## Setup and run
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                   # open http://127.0.0.1:5000
```

## Using several models
- Copy your `.keras` files into `models/` and click **Refresh**, or use **Add model file** on the page.
- Tick one or more models, add one or more images, and click **Run prediction**.
- Each image gets one row per model, plus a note if the models agree or disagree.
- **Download CSV** exports all results for your own comparison.
- Only `MAX_LOADED` models (default 2) stay in memory at once; others are unloaded automatically,
  so 7 to 8 large models won't exhaust RAM. Running many models at once is slower because each one loads in turn.

## Preprocessing (important)
Each model must be preprocessed the way it was trained. Set it per model in `models/models.json`:
```json
{
  "my_fusion_v1.keras": { "preprocess": "rescale", "resize": "bilinear", "threshold": 0.5 },
  "my_fusion_v2.keras": { "preprocess": "tf" }
}
```
| `preprocess` | Pixel range | Use when |
|---|---|---|
| `rescale` | 0 to 1 | trained with `/255` (default) |
| `tf` | -1 to 1 | used `resnet_v2` / `mobilenet_v2` `preprocess_input` |
| `none` | 0 to 255 | preprocessing layers are inside the model |

The **Preprocessing** dropdown on the page overrides this for one run, which is handy for finding
the right setting for a model. Global defaults can also be set with the `PREPROCESS`, `RESIZE`,
`THRESHOLD`, `MAX_LOADED`, `MODELS_DIR` and `PORT` environment variables.

## Output handling
- Single sigmoid output: treated as P(class 1 = Non-ASD). Two softmax outputs: index 1 is Non-ASD.
- Two-input fusion models: the same image is fed to both inputs.
- Confidence = probability of the predicted class.

Only load `.keras` files you trust: loading a model can run code stored in it.
For research and model testing only, not a diagnostic tool.
