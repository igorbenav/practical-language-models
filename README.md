# Practical Language Models

From Intuition to Agents in Production.

> **[Read the English version (PDF)](docs/en/Practical-Language-Models.pdf)**

A book that teaches language models starting from the learning problem and neural networks, through word embeddings and transformers, all the way to building LLM applications, agents, and deploying them to production. Opinionated, hands-on, and meant to be read in order. Each chapter builds on the previous one.

Assumes Python knowledge. No prior ML experience needed.

## Table of Contents

### Part I: Foundations

| Chapter | Topic | Code |
|---------|-------|------|
| 1 | The Learning Problem | [chapter1](en/code/chapter1) |
| 2 | Learning to Learn | [chapter2](en/code/chapter2) |
| 3 | From Words to Numbers | [chapter3](en/code/chapter3) |
| 4 | Attention and Transformers | |
| 5 | Building a Language Model | |

### Part II: Building with LLMs

| Chapter | Topic | Code |
|---------|-------|------|
| 6 | What LLMs Can and Can't Do | |
| 7 | Your First LLM Application | |
| 8 | Tools: Letting Agents Act | |
| 9 | Structured Output | |
| 10 | Retrieval-Augmented Generation | |
| 11 | Fine-Tuning | |
| 12 | Multi-Step Workflows and Pipelines | |

### Part III: Deploying to Production

| Chapter | Topic | Code |
|---------|-------|------|
| 13 | Serving with FastAPI | |
| 14 | Making It Reliable | |
| 15 | Deployment | |

More chapters will be added.

## Available Languages

| Language | Directory | Status |
|----------|-----------|--------|
| English | [en/](en/) | In progress |

## Prerequisites

- [Python 3.11+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager
- [Quarto](https://quarto.org/docs/get-started/) - Publishing system

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/igorbenav/practical-language-models.git
   cd practical-language-models
   ```

2. Install Python dependencies with uv:
   ```bash
   uv sync
   ```

## Building the Book

### HTML (default)

```bash
uv run quarto render en/
```

The output will be in `docs/en/`. Open `docs/en/index.html` in your browser to view.

### PDF

```bash
uv run quarto render en/ --to pdf
```

> Note: PDF rendering requires a LaTeX distribution (e.g., [TinyTeX](https://yihui.org/tinytex/), [TeX Live](https://www.tug.org/texlive/), or [MiKTeX](https://miktex.org/)).

### Preview with live reload

```bash
uv run quarto preview en/
```

This starts a local server and opens the book in your browser. Changes to `.qmd` files will automatically trigger a rebuild.

## Project Structure

```
practical-language-models/
├── en/                          # English
│   ├── _quarto.yml
│   ├── index.qmd
│   ├── references.qmd
│   ├── references.bib
│   ├── chapters/
│   │   ├── chapter1.qmd
│   │   ├── chapter2.qmd
│   │   └── chapter3.qmd
│   ├── images/                  # Figures (language-specific)
│   │   ├── chapter1/
│   │   ├── chapter2/
│   │   └── chapter3/
│   └── code/                    # Completed code per chapter
│       ├── chapter1/
│       ├── chapter2/
│       └── chapter3/
├── docs/                        # Rendered output
│   └── en/
├── README.md
├── LICENSE
├── pyproject.toml
└── .gitignore
```

## Adding a New Language

To translate the book into a new language:

1. Copy the `en/` directory as your starting point:
   ```bash
   cp -r en/ pt/   # for Portuguese, for example
   ```

2. In your new directory (e.g., `pt/`):
   - Translate all `.qmd` files (chapters, index, references)
   - Translate or recreate images that contain text (SVGs in `images/`)
   - Update `_quarto.yml`: change `output-dir` to `../docs/pt`, translate the title/subtitle, and update `output-file` to use your language code (e.g., `"Practical Language Models (pt)"`)
   - Code in `code/` may need translated comments

3. Update the cover image:
   - Edit `images/cover.svg` translate the text elements (title, subtitle, author) to your language
   - Convert the SVG to PNG for PDF rendering (requires `rsvg-convert`, installed via `librsvg`):
     ```bash
     rsvg-convert -f png -w 2400 -h 3600 -o pt/images/cover.png pt/images/cover.svg
     ```

4. Build with:
   ```bash
   uv run quarto render pt/
   uv run quarto render pt/ --to pdf
   ```

5. Add yourself to the contributors table below and submit a pull request.

## Contributors

| Name | GitHub | Role | Language |
|------|--------|------|----------|
| Igor Benav | [@igorbenav](https://github.com/igorbenav) | Author | English |

When your translation or review is merged, add yourself in the same pull request to:

1. The **Contributors** table above (this README)
2. The **Contributors** table in your language's `index.qmd` (this appears in the actual book)

Roles: "Translator", "Reviewer", or both.

## License

MIT License - see [LICENSE](LICENSE) for details.
