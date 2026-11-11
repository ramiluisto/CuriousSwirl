# Antonyms and Synonyms: Geometric Structure in Word Embedding Difference Vectors

**This repo has been generated/summarized from a larger project repo by LLM-based coding assistants. Users beware.**

---

Companion code for the paper [*"A visual observation on the geometry of UMAP projections of the difference vectors of antonym and synonym word pair embeddings"*](https://arxiv.org/abs/2603.24150).

This repository reproduces the UMAP projection visualizations showing a consistent "swirl" pattern in the difference vectors of synonym and antonym word embeddings — stable across Word2Vec, GloVe, BERT, and OpenAI embedding models.

## Quick Start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/run_all.py                    # Core reproducible pipeline
python scripts/generate_paper_figures.py     # Regenerate the paper's key figures
python scripts/generate_cosine_sim_plots.py  # Regenerate cosine similarity violins
```

The pipeline produces intermediate data (embeddings, pairs, statistics, projections, classification demo) in `results/`. The figure-generation scripts produce the key visual results — including the "swirl" — and save them to `results/images/`. The static paper figures are included in `prose/img/` for reference.

## First-Run Cost

The first successful full run is dominated by model downloads and CPU-bound projection steps.

- **Python dependencies**: `uv pip install -e ".[dev]"` is usually quick, but will download PyTorch and the scientific Python stack. On our test run, the largest package download was PyTorch at about 77 MB.
- **Model/data downloads for the full pipeline**:
  - BERT base-cased: about 400 MB
  - Word2Vec Google News: about 1.7 GB
  - GloVe 6B zip: 822 MB manual download
- **Approximate network traffic for a full first run**: about 3.0 GB for model assets, plus Python package downloads.
- **Approximate disk impact**: the GloVe archive extracts into a large text file, so plan for roughly 1 GB extra disk usage beyond the zip itself, plus space for generated results.
- **Typical wall-clock time**:
  - Dependency install: usually 1-5 minutes
  - Core pipeline (`run_all.py`): roughly 45-90 minutes on a modern laptop CPU after downloads are in place. On our test machine (Apple M4 Max) the pipeline took about 60 minutes.
  - Figure generation adds roughly 30 minutes (dominated by UMAP projections with various hyperparameters).
  - On a slow connection or cold caches, the first run can easily exceed two hours.

The repository already includes the expensive OpenAI embeddings and the LLM validation files, so those are not downloaded during the normal reproduction flow.

## Models

| Model | Dimensions | Source |
| ----- | ---------- | ------ |
| BERT base-cased | 768 | HuggingFace (auto-download) |
| Word2Vec | 300 | gensim (auto-download, ~1.7 GB) |
| GloVe 6B | 300 | Manual download (see below) |
| text-embedding-3-small | 1536 | Pre-computed (included) |
| text-embedding-3-large | 3072 | Pre-computed (included) |

## Pipeline

### Step 01: Data preparation

```bash
python scripts/01_prepare_data.py        # Load Stuttgart synonym/antonym dataset
python scripts/01a_filter_single_token.py # Filter to BERT single-token pairs
```

### Step 02: Extract embeddings

```bash
python scripts/02_extract_embeddings.py
```

Auto-downloads BERT (~400 MB) and word2vec (~1.7 GB) on first run. OpenAI embeddings are pre-included. If GloVe is not installed locally, the extraction step skips it with a warning.

### Step 03: Generate pairs

```bash
python scripts/03_generate_pairs.py
```

Generates 4 pair types per model: synonyms, antonyms, shuffled synonym words, shuffled antonym words. Uses the pre-computed LLM validation results in `results/validation/`.

### Step 04: Compute statistics

```bash
python scripts/04_compute_statistics.py
```

Per-pair cosine similarity, Euclidean distance, and other metrics.

### Step 05: Classification demo

```bash
python scripts/05_classify_demo.py
```

Demonstrates antonym-vs-synonym classification with Logistic Regression and a Shallow Neural Network on two models, using both random and word-aware (lexical) splits. Results are saved to `results/classification/`. The full classification results in the paper were produced with additional methods and hyperparameter sweeps beyond what this demo covers.

### Step 06: Compute projections

```bash
python scripts/06_compute_projections.py
```

PCA, t-SNE, and UMAP projections of the difference and concatenation vectors.

### Generate paper figures

```bash
python scripts/generate_paper_figures.py      # Figures 2-6: UMAP/PCA/t-SNE scatter grids
python scripts/generate_cosine_sim_plots.py   # Figure 1: cosine similarity violin plots
```

These scripts produce the paper's key visual results (including the "swirl") and save them to `results/images/`. The static paper figures are included in `prose/img/` for reference. To regenerate the full five-model figure set, install GloVe first — several paper panels depend on it.

## Row-Order Controls (read before interpreting the swirl)

A later analysis of this repository found that the swirl follows the **row order of
the data matrix**: umap-learn's default layout optimiser (0.5.x, and still the
default in 0.6.0) visits the graph edges in row order, and on the hub-dominated
neighbour graph of the difference vectors this lays the points out along a curve
traced in storage order. Because the
four pair types are stacked as four row blocks, they appear as separate coloured
filaments. Shuffling the rows of the same matrix keeps the filament and mixes the
colours; i.i.d. noise with unequal norms produces the same picture. See
[`analysis/README.md`](./analysis/README.md) for the evidence, the consequences for
the classification tables, a projection-free analysis of what the embeddings do
show, and a review of related literature.

To explore this yourself, open
[`analysis/explorer/swirl_explorer.html`](./analysis/explorer/swirl_explorer.html) in a
browser: a self-contained page with every model's layout under several row orderings
(as stored, shuffled, sorted alphabetically, ...) and colourings.

```bash
python scripts/07_row_order_controls.py          # permuted rows, shuffled edge order, noise nulls
python scripts/08_projection_free_analysis.py    # probes and second-moment contrast, no projection
python scripts/generate_hp_grid.py --model bert-base-cased --shuffle-rows
python scripts/09_build_explorer.py              # data for the interactive page
quarto render analysis/explorer/swirl_explorer.qmd
```

The default figure scripts are unchanged and still reproduce the figures of the
paper.

## Generating Custom UMAP Grids

You can generate UMAP hyperparameter grids for any model using the dedicated script:

```bash
# Default grid for Word2Vec (all four pair types)
python scripts/generate_hp_grid.py --model word2vec

# BERT with only antonyms and synonyms (no shuffled controls)
python scripts/generate_hp_grid.py --model bert-base-cased \
    --pair-types antonyms synonyms

# Custom hyperparameter ranges for OpenAI Large
python scripts/generate_hp_grid.py --model text-embedding-3-large \
    --nn 15 30 50 --md 0.01 0.1 0.5

# Save to a custom directory
python scripts/generate_hp_grid.py --model glove --output-dir my_figures/

# Row-order control: fit UMAP on shuffled rows (written to a separate file)
python scripts/generate_hp_grid.py --model bert-base-cased --shuffle-rows
```

Available models: `word2vec`, `glove`, `bert-base-cased`, `text-embedding-3-small`, `text-embedding-3-large`.

## GloVe Setup

GloVe embeddings must be downloaded manually:

1. Download `glove.6B.zip` (822 MB) from [the Stanford GloVe page](https://nlp.stanford.edu/projects/glove/)
2. Extract `glove.6B.300d.txt`
3. Place at `data/glove/glove.6B.300d.txt` or `~/.cache/glove/glove.6B.300d.txt`

The default `scripts/run_all.py` pipeline skips GloVe when the file is absent. However, the full five-model paper reproduction, including regeneration of the paper figures, still requires GloVe to be installed.

## Pre-Computed Data

This repository includes data that would be expensive to regenerate:

- **OpenAI embeddings** (`data/OpenAI_embeddings/`): Extracted via the OpenAI API. The large model embeddings are in compressed `.npz` format.
- **LLM validation results** (`results/validation/`): Each word pair rated by 4 LLM judges on a 1-5 scale. Pairs below 3.0 filtered out.

## Compiling the Paper

Requires a LaTeX distribution with `pdflatex` and standard packages (amsmath, graphicx, booktabs, tikz, etc.).

```bash
cd prose
pdflatex HeyLookAtThisSwirl
bibtex HeyLookAtThisSwirl
pdflatex HeyLookAtThisSwirl
pdflatex HeyLookAtThisSwirl
```

## Running Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```

## Citation

```bibtex
@article{luisto2026swirl,
  title={A visual observation on the geometry of UMAP projections of the difference vectors of antonym and synonym word pair embeddings},
  author={Luisto, Rami},
  journal={arXiv preprint arXiv:2603.24150},
  year={2026}
}
```

## License

[MIT](./LICENSE)
