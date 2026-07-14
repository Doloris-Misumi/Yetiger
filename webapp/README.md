# YesTiger Web Studio

Local web wrapper for the current YesTiger prototype:

```text
audio upload
-> allin1 downbeat grid
-> frozen MuQ seed46 bar-level coarse structure model
-> support action recommender
-> synchronized canvas timeline / callbook export
```

The older LOSO/tiny-pipeline web app has been compressed under:

```text
webapp/_archive/legacy_loso_webapp_20260630.zip
```

## Run

```powershell
.\train\venv\Scripts\python.exe webapp\server.py --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

## Inputs

- Example songs are loaded from `support/recommendations/paper_test/`.
- Uploaded audio is analyzed with `train/frozen_candidate.json`.
- Feature upload caches and generated jobs are written under `webapp_runs/`.

## Outputs

Each uploaded job saves:

```text
webapp_runs/jobs/<job_id>/result.json
webapp_runs/jobs/<job_id>/prediction.json
webapp_runs/jobs/<job_id>/support.json
webapp_runs/jobs/<job_id>/callbook.md
```

The browser can also export edited JSON, Markdown callbooks, and recorded WebM
previews from the canvas.
