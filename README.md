# Academic Paper OCR & Structuring Pipeline

## Overview

This project implements an automated pipeline for extracting and structuring text from academic papers (or similar documents), typically obtained in PDF format. The core goal is to make the content of these documents machine-readable for further analysis, data mining, or indexing.

It addresses the challenges of inaccessible text in image-based or complex PDF layouts, the manual effort required for large-scale digitization, and the need for consistent, structured output from diverse sources.

## Features

- Automated acquisition of PDF documents (via scraping module).
- Conversion of PDF pages to images for OCR processing.
- High-accuracy Optical Character Recognition (OCR) using the GOT-OCR2.0 model.
- Post-OCR text cleaning to improve quality.
- Structuring of extracted text into JSON format, potentially including layout metadata.
- Containerized environment for consistent setup and execution.
- GPU acceleration support for efficient OCR processing.

## Pipeline Architecture

The system operates as a multi-stage processing pipeline:

1.  **Data Acquisition (Scraping)**: PDF documents are fetched (e.g., by `src/scrapper/paper_scrape.py`) and stored (e.g., in `data/pdfs/`).
2.  **Preprocessing (PDF to Image)**: PDFs are converted into PNG images page by page (e.g., by `src/ocr/pdf_to_png.py`), with outputs to a directory like `data/images/`.
3.  **OCR Execution**: The GOT-OCR2.0 engine processes these images to extract text, orchestrated by scripts like `src/ocr/multi_page_run_ocr_2.0.py`. Raw text might be stored in `data/raw_text/`.
4.  **Postprocessing (Text Cleaning)**: The raw OCR output is refined (e.g., by `src/process_pdf/clean_text.py`), with cleaned text potentially in `data/processed_text/`.
5.  **Structuring (JSON Output)**: Cleaned text is formatted into JSON (e.g., by `src/process_pdf/create_json.py`, `src/process_pdf/combine_json.py`), with final outputs in `data/processed_json/` and potentially an aggregated file like `data/all_processed_data.json`.
6.  **(Potential) Evaluation**: Metrics on pipeline output quality may be calculated (related to `data/metrics/`).

## Technology Stack

- **Containerization**: Docker (using `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` base image)
- **Programming Language**: Python 3.10
- **Core OCR Engine**: GOT-OCR2.0 (from `https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git`)
- **Key Python Libraries**:
  - `transformers`
  - `flash-attn`
  - `pdf2image`
  - Standard Python build/packaging tools (`pip`, `setuptools`, etc.)
- **System Dependencies (via Docker)**: `git`, `poppler-utils`, CUDA libraries.

## Project Structure

```
final_project/
├── Dockerfile              # Defines the containerized environment
├── README.md               # This file
├── data/                   # Input, intermediate, and output data
│   ├── pdfs/
│   ├── images/
│   ├── raw_text/
│   ├── processed_text/
│   ├── processed_json/
│   └── metrics/
├── memory-bank/            # Project context and documentation (see below)
│   ├── projectbrief.md
│   ├── productContext.md
│   ├── techContext.md
│   ├── systemPatterns.md
│   ├── activeContext.md
│   └── progress.md
└── src/                    # Custom Python scripts for pipeline stages
    ├── scrapper/
    │   └── paper_scrape.py
    ├── ocr/
    │   ├── pdf_to_png.py
    │   └── multi_page_run_ocr_2.0.py
    └── process_pdf/
        ├── clean_text.py
        ├── create_json.py
        └── combine_json.py
```

## Setup and Installation

The project is designed to run within a Docker container.

1.  **Prerequisites**:
    - Docker installed.
    - NVIDIA GPU with appropriate drivers if CUDA acceleration is desired (the Docker image is CUDA-enabled).
2.  **Build the Docker Image**:
    ```bash
    docker build -t got-ocr-pipeline .
    ```
3.  **GOT-OCR2.0 Model Weights**: The `Dockerfile` creates a `/GOT_weights` directory. The GOT-OCR2.0 model requires pre-trained weights. Refer to the [GOT-OCR2.0 repository](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) for instructions on obtaining these weights and ensure they are correctly mounted or placed into this directory within the container, or modify the run scripts to download them if applicable.

## Running the Pipeline

The exact command to run the full pipeline will depend on how the scripts in `src/` are designed to be invoked and how data is passed between them. The `Dockerfile` default command is `CMD ["python3", "GOT/demo/run_ocr_2.0.py", "--help"]`, which runs the demo from the cloned GOT-OCR2.0 repository.

To run custom pipeline stages, you would typically use `docker run` with the appropriate command:

```bash
# Example: Running a specific script (you'll need to adjust volume mounts)
# docker run --gpus all -v $(pwd)/data:/app/data got-ocr-pipeline python3 src/ocr/multi_page_run_ocr_2.0.py <args>
```

Refer to the scripts within the `src/` directory for specific arguments and execution details. The overall flow involves:

1. Placing input PDFs into `data/pdfs/`.
2. Running the sequence of scripts for conversion, OCR, cleaning, and JSON generation.
3. Checking `data/processed_json/` for outputs.

## Memory Bank

This project uses a `memory-bank/` directory to store detailed, evolving documentation about its context, architecture, progress, and technical details. This is used by AI assistants to maintain continuity and understanding of the project over time. For more details on its structure and purpose, refer to the `ai_instructions.md` (if available, or the general concept outlined by the AI assistant).
