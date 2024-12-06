# Machine Learning Study Notes Generator

This project provides a automated pipeline for processing and generating comprehensive machine learning study notes from lecture transcripts. It includes tools for chunking lecture content, generating detailed notes using LLMs, and combining them into a single PDF document.

## Project Structure

```
.
├── 00_chunk.py               # Script for chunking lecture transcripts
├── 01_make_notes.py         # Script for generating notes using LLM
├── 02_combine.py            # Script for combining notes into PDF
├── CS7641 ML Lectures All Chapters Raw.txt  # Raw lecture transcript
├── README.md                # This file
├── lecture_notes/           # Generated markdown notes and final PDF
└── lecture_scripts/         # Chunked lecture transcripts
```

## Features

- **Transcript Chunking**: Breaks down lecture transcripts into manageable sections
- **Automated Note Generation**: Uses LLM to create detailed, graduate-level notes
- **PDF Compilation**: Combines all notes into a single, well-formatted PDF document
- **Hierarchical Organization**: Maintains chapter and section structure from source material

## Prerequisites

- Python 3.x
- pandoc (for PDF generation)
- LaTeX distribution (for PDF generation)
- Required Python packages:
  ```
  llama_index
  python-dotenv
  ```

## Setup

1. Clone the repository
2. Install required packages:
   ```bash
   pip install llama_index python-dotenv
   ```
3. Install pandoc and a LaTeX distribution
4. Create a `.env` file with your Azure OpenAI credentials:
   ```
   MODEL_NAME=your_model_name
   ENGINE=your_engine
   ```

## Usage

### 1. Chunk Lecture Transcript

```bash
python 00_chunk.py
```
This splits the main transcript file into section-level files in `lecture_scripts/`.

### 2. Generate Notes

```bash
python 01_make_notes.py
```
This processes each section and generates detailed markdown notes in `lecture_notes/`. 

### 3. Create PDF

```bash
python 02_combine.py
```
This combines all markdown files into a single PDF with proper formatting. Note that OpenAI's LLM cannot maintain a consistent structure across sections, therefore it may cause the pdf formatting to be inconsistent. 

## Note Generation Format

The generated notes follow a structured format:

1. Theoretical Foundations
   - Core mathematical principles
   - Formal definitions
   - Key theorems and proofs

2. Key Concepts and Methodology
   - Essential concepts
   - Algorithms and methods
   - Implementation details

3. Applications and Case Studies
   - Practical examples
   - Implementation variations
   - Performance considerations

4. Key Takeaways and Exam Focus
   - Critical points
   - Common exam topics
   - Important equations

## Contributing
I created this note generator for my personal use over a short period, so there are many potential improvements, such as prompt fine-tuning, automatic formatting, and more.
Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE)