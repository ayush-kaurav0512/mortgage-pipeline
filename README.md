# axia-pipeline

A mortgage document pipeline that takes raw PDF loan documents, extracts and normalizes their content, runs a set of rule-based checks, and produces a deal scorecard summarizing the health of each deal.

## Folder structure

- **input/** — Drop raw mortgage PDFs (loan applications, appraisals, income docs, title work, etc.) into this folder. The pipeline reads from here.
- **parsed/** — Holds one structured JSON file per input document after the parser and normalizer have run. These are the clean, machine-readable versions of the PDFs.
- **flags/** — Holds JSON flag reports produced by the flag engine. Each report lists inconsistencies, missing data, and risk signals found across a deal's documents.
- **reports/** — Holds the final deal scorecards. Each scorecard summarizes the deal at a glance: key facts, flags raised, and an overall assessment.

## Source files (`src/`)

- **parser.py** — Opens each PDF in `input/` and pulls out the raw text and tables using `pdfplumber`. Think of it as the "read the PDF" step. It does not try to understand the content — it just gets it off the page.
- **normalizer.py** — Takes the raw text from the parser and turns it into a clean, structured JSON object with consistent field names (borrower name, loan amount, property address, etc.). Uses an LLM via the OpenAI API to handle the messy variation between document formats. Output lands in `parsed/`.
- **flag_engine.py** — Loads the normalized JSON for a deal and runs a set of rules over it: do the borrower names match across documents? Is the income consistent? Are required fields missing? Anything that looks off becomes a flag, and the full flag list is written to `flags/`.
- **main.py** — The entry point. Runs the whole pipeline end-to-end: parse → normalize → flag → scorecard. This is the file you actually run to process a batch of documents.

## Configuration

- **.env.example** — Template for your environment variables. Copy it to `.env` and fill in your `OPENAI_API_KEY`.
- **requirements.txt** — Python dependencies for the project.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your OpenAI key
# drop PDFs into input/
python src/main.py
```
