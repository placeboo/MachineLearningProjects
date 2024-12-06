"""
In these file, we will combine all the markdown files into one signle file and save it as a pdf file.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import re


def extract_title(content):
    """
    Extract title from markdown content handling different formats.
    Matches: 'TITLE:', '#TITLE:', or '# TITLE:'

    Args:
        content (str): Markdown content

    Returns:
        str: Extracted title or None if not found
    """
    # Try different possible title formats
    patterns = [
        r'(?:^|\n)\*\*\s*TITLE:\s*(.+?)\*\*(?:\n|$)',  # Matches **TITLE: title**
        r'(?:^|\n)#{1,6}\s*TITLE:\s*(.+?)(?:\n|$)',  # Matches # TITLE:, ## TITLE:, etc.
        r'(?:^|\n)#?\s*TITLE:\s*(.+?)(?:\n|$)',  # Matches TITLE: or # TITLE:
        r'(?:^|\n)#\s*TITLE\s*(.+?)(?:\n|$)',  # Matches #TITLE
        r'(?:^|\n)TITLE\s*(.+?)(?:\n|$)',  # Matches TITLE without colon
        r'(?:^|\n)\*\*\s*TITLE\s*(.+?)\*\*(?:\n|$)'  # Matches **TITLE title**
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)  # Make it case insensitive
        if match:
            # Clean up the extracted title
            title = match.group(1).strip()
            # Remove any trailing colons or hashes
            title = re.sub(r'[:#\s]+$', '', title)
            return title

    return None


def clean_content(content):
    """
    Remove the original title line from the content.

    Args:
        content (str): Original markdown content

    Returns:
        str: Content with title line removed
    """
    patterns = [
        r'(?:^|\n)\*\*\s*TITLE:\s*(.+?)\*\*(?:\n|$)',  # Matches **TITLE: title**
        r'(?:^|\n)#{1,6}\s*TITLE:\s*(.+?)(?:\n|$)',  # Matches # TITLE:, ## TITLE:, etc.
        r'(?:^|\n)#?\s*TITLE:\s*(.+?)(?:\n|$)',  # Matches TITLE: or # TITLE:
        r'(?:^|\n)#\s*TITLE\s*(.+?)(?:\n|$)',  # Matches #TITLE
        r'(?:^|\n)TITLE\s*(.+?)(?:\n|$)',  # Matches TITLE without colon
        r'(?:^|\n)\*\*\s*TITLE\s*(.+?)\*\*(?:\n|$)'  # Matches **TITLE title**
    ]

    cleaned_content = content
    for pattern in patterns:
        cleaned_content = re.sub(pattern, '\n', cleaned_content, flags=re.IGNORECASE)

    return cleaned_content.strip()


def read_file_safely(file_path):
    """
    Attempts to read a file using different encodings.

    Args:
        file_path (str): Path to the file to read

    Returns:
        str: Content of the file

    Raises:
        RuntimeError: If file cannot be read with any encoding
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Could not read file {file_path} with any of the attempted encodings: {encodings}")


def combine_markdown_files(input_dir, output_file):
    """
    Combines multiple markdown files into a single PDF while preserving formatting.
    """
    try:
        # Check if pandoc is installed
        subprocess.run(['pandoc', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pandoc is not installed. Please install pandoc first.")
        return

    # Get all markdown files and sort them
    md_files = []
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.md'):
            md_files.append(os.path.join(input_dir, file))

    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    # Create a temporary file to store combined markdown
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
        # Add YAML metadata with enhanced formatting
        temp_file.write("""---
title: "Machine Learning Notes"
date: \\today
author: "Jackie Yin"
geometry: margin=1in
fontsize: 12pt
header-includes: |
    \\usepackage{fancyhdr}
    \\pagestyle{fancy}
    \\usepackage{indentfirst}
    \\usepackage{amsmath}
    \\usepackage{amssymb}
    \\usepackage{listings}
    \\usepackage{enumitem}
    \\usepackage{tcolorbox}
    \\usepackage{fancyvrb}
    \\usepackage{xcolor}
    \\usepackage{etoolbox}

    \\definecolor{light-gray}{gray}{0.95}
    \\lstset{
        basicstyle=\\ttfamily\\small,
        breaklines=true,
        frame=single,
        backgroundcolor=\\color{light-gray},
        tabsize=2,
        showstringspaces=false,
        breakindent=0pt,
        keepspaces=true
    }

    \\setlistdepth{9}
    \\setlist[itemize,1]{label=$\\bullet$}
    \\setlist[itemize,2]{label=$\\circ$}
    \\setlist[itemize,3]{label=$\\diamond$}
    \\setlist{nosep}
    \\setlist[itemize]{leftmargin=*}
    \\setlist[enumerate]{leftmargin=*}

    \\BeforeBeginEnvironment{verbatim}{\\begin{tcolorbox}[colback=light-gray]}
    \\AfterEndEnvironment{verbatim}{\\end{tcolorbox}}
---

""")

        # Process each markdown file
        for i, md_file in enumerate(md_files, 1):
            print(f"Processing {md_file}...")

            # Add a page break before each new section (except the first one)
            if i > 1:
                temp_file.write("\n\\newpage\n\n")

            try:
                # Read content and extract title
                content = read_file_safely(md_file)
                title = extract_title(content)

                if title:
                    # Write chapter title
                    temp_file.write(f"# {title}\n\n")
                    # Clean content by removing original title line
                    content = clean_content(content)
                else:
                    print(f"Warning: No title found in {md_file}")

                # Write remaining content
                temp_file.write(content)
                temp_file.write("\n\n")

            except Exception as e:
                print(f"Error processing file {md_file}: {str(e)}")
                continue

    # Update pandoc command with specific markdown extensions
    try:
        subprocess.run([
            'pandoc',
            temp_file.name,
            '-o', output_file,
            '--pdf-engine=xelatex',
            '--highlight-style=tango',
            '--toc',
            '--number-sections',
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=blue',
            '-V', 'urlcolor=blue',
            '--standalone',
            '--from', 'markdown+raw_tex+tex_math_dollars+tex_math_single_backslash+lists_without_preceding_blankline',
            '--wrap=preserve',
            '--columns=72',
            '--variable=verbatim-in-note',
            '--top-level-division=chapter'
        ], check=True)
        print(f"Successfully created PDF: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting to PDF: {e}")
        # Print the detailed error message if available
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")
    finally:
        os.unlink(temp_file.name)


def main():
    # Configure input and output paths
    input_dir = "lecture_notes"  # Directory containing markdown files
    output_file = "machine_learning_notes.pdf"

    # Create output directory if it doesn't exist
    output_file = os.path.join(input_dir, output_file)
    combine_markdown_files(input_dir, output_file)


if __name__ == "__main__":
    main()