from pathlib import Path
import subprocess
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
R_SCRIPT = BASE_DIR / "tfidf_rag.R"

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"error": "Please enter a question."}), 400

    if not R_SCRIPT.is_file():
        return jsonify({
            "error": f"R script not found: {R_SCRIPT}. Put tfidf_rag.R beside app.py."
        }), 500

    try:
        # Pass the question as one argument. subprocess.run does not use a shell.
        result = subprocess.run(
            ["Rscript", str(R_SCRIPT), query],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
    except FileNotFoundError:
        return jsonify({
            "error": "Rscript was not found. Install R and make sure the 'Rscript' command is on PATH."
        }), 500
    except subprocess.TimeoutExpired:
        return jsonify({
            "error": "The R/Ollama request timed out after 5 minutes. Check whether Ollama is running and the model is loaded."
        }), 504
    except Exception as exc:
        app.logger.exception("Failed to run R script")
        return jsonify({"error": f"Could not execute R script: {exc}"}), 500

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if result.returncode != 0:
        detail = stderr or stdout or f"Rscript exited with code {result.returncode}."
        return jsonify({
            "error": "The R script failed.",
            "details": detail,
            "returncode": result.returncode,
        }), 500

    # The current R script prints its answer and retrieved passages as terminal text.
    # Return that output as-is instead of requiring changes to tfidf_rag.R.
    return jsonify({
        "output": stdout or "R script finished without producing output.",
        "stderr": stderr,
        "returncode": result.returncode,
    })


if __name__ == "__main__":
    # Local development server. Keep it on localhost unless you add authentication.
    app.run(host="127.0.0.1", port=5000, debug=False)
