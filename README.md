# Antonyms and Synonyms: Geometric Structure in Word Embedding Difference Vectors

Companion code for the paper *"Hey look at this cool swirl that appears in a sort of specific projection of the difference vectors of the embedding vectors of antonym and synonym pairs!"*

This repository reproduces the UMAP projection visualizations showing a consistent "swirl" pattern in the difference vectors of synonym and antonym word embeddings — stable across Word2Vec, GloVe, BERT, and OpenAI embedding models.

## Quick Start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/run_all.py
python scripts/generate_paper_figures.py
python scripts/generate_cosine_sim_plots.py
```

## Models

| Model | Dimensions | Source |
|-------|-----------|--------|
| BERT base-cased | 768 | HuggingFace (auto-download) |
| Word2Vec | 300 | gensim (auto-download, ~1.7 GB) |
| GloVe 6B | 300 | Manual download (see below) |
| text-embedding-3-small | 1536 | Pre-computed (included) |
| text-embedding-3-large | 3072 | Pre-computed (included) |

## Pipeline

### 1. Data preparation
```bash
python scripts/01_prepare_data.py        # Load Stuttgart synonym/antonym dataset
python scripts/01a_filter_single_token.py # Filter to BERT single-token pairs
```

### 2. Extract embeddings
```bash
python scripts/02_extract_embeddings.py
```
Auto-downloads BERT (~400 MB) and word2vec (~1.7 GB) on first run. OpenAI embeddings are pre-included. For GloVe, see below.

### 3. Generate pairs
```bash
python scripts/03_generate_pairs.py
```
Generates 4 pair types per model: synonyms, antonyms, shuffled synonym words, shuffled antonym words. Uses the pre-computed LLM validation results in `results/validation/`.

### 4. Compute statistics
```bash
python scripts/04_compute_statistics.py
```
Per-pair cosine similarity, Euclidean distance, and other metrics.

### 5. Compute projections
```bash
python scripts/06_compute_projections.py
```
PCA, t-SNE, and UMAP projections of the difference and concatenation vectors.

### 6. Generate paper figures
```bash
python scripts/generate_paper_figures.py      # Figures 2-6: UMAP/PCA/t-SNE scatter grids
python scripts/generate_cosine_sim_plots.py   # Figure 1: cosine similarity violin plots
```

Pre-generated figures are included in `prose/img/`.

## GloVe Setup

GloVe embeddings must be downloaded manually:

1. Download `glove.6B.zip` (822 MB) from https://nlp.stanford.edu/projects/glove/
2. Extract `glove.6B.300d.txt`
3. Place at `data/glove_file/glove.6B.300d.txt` or `~/.cache/glove/glove.6B.300d.txt`

The pipeline will skip GloVe if the file is not found.

## Pre-Computed Data

This repository includes data that would be expensive to regenerate:

- **OpenAI embeddings** (`data/OpenAI_embeddings/`): Extracted via the OpenAI API. The large model embeddings are in compressed `.npz` format.
- **LLM validation results** (`results/validation/`): Each word pair rated by 4 LLM judges on a 1-5 scale. Pairs below 3.0 filtered out.

## Compiling the Paper

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
@article{luisto2025swirl,
  title={Hey look at this cool swirl that appears in a sort of specific projection
         of the difference vectors of the embedding vectors of antonym and synonym pairs!
         It seems to be stable across different model types so maybe it is not just
         a projection artefact but represents some real structure?},
  author={Luisto, Rami},
  year={2025}
}
```

## License

MIT
