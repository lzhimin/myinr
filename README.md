# INR — Neural Networks for 3D Scientific Data Modeling

A collection of neural network models designed for 3D scientific data modeling. This project explores implicit neural representations (INR) and related architectures for learning, reconstructing, and querying volumetric scientific datasets.

## Overview

3D scientific data (e.g. medical volumes, simulation fields, geospatial data) is often large, irregular, and hard to represent efficiently. This project provides a set of neural models that can:

- Learn compact representations of 3D volumetric data
- Reconstruct continuous fields from sparse or noisy measurements
- Query 3D data at arbitrary resolutions

## Project Structure

```
inr/
├── data/          # 3D datasets (not tracked by git)
├── src/inr/       # Source code
├── tests/         # Unit tests
└── requirements.txt
```

## Getting Started

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run
python src/inr/main.py
```

## Models

Models will be added progressively. Planned architectures include:

- **SIREN** — Sinusoidal representation networks
- **NeRF** — Neural radiance fields
- **Instant-NGP** — Hash-encoded neural fields

## Data

Place your 3D datasets in the `data/` folder. This folder is excluded from version control.
