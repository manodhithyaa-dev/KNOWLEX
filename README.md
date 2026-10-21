# HTML + CSS + Python subprocess frontend for tfidf_rag.R

## Files

```text
project/
├── app.py
├── tfidf_rag.R        # Put your existing R script here, unchanged
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

Your R script should remain beside `app.py`, and the `knowledge/` directory should be in the same working directory as your R script expects.

## Setup on Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the R packages used by your script if you have not already:

```bash
Rscript -e 'install.packages("httr", repos="https://cloud.r-project.org")'
```

Make sure Ollama is running and your configured model has been pulled. Then start Flask:

```bash
python app.py
```

Open http://127.0.0.1:5000

## How it works

The browser posts the question to `/api/ask`. Flask calls:

```bash
Rscript /absolute/path/to/tfidf_rag.R "the question"
```

Python captures stdout and stderr and returns stdout to the browser. The R script's current terminal output is displayed as plain text; this does not change your R retrieval logic.

Note: The existing R script loads `knowledge/` relative to its working directory, so `app.py` runs the subprocess with the project folder as `cwd`. It also has a 300-second timeout for Ollama generation.
