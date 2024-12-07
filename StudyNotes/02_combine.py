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


def update_yaml_header():
    """
    Return an enhanced YAML header with LaTeX settings for code blocks with math.
    """
    return """---
title: "Machine Learning Notes"
author: "Jackie Yin"
date: \\today
documentclass: report
geometry: "margin=1in"
fontsize: 12pt
header-includes:
    - \\usepackage{fancyhdr}
    - \\pagestyle{fancy}
    - \\usepackage{amsmath}
    - \\usepackage{amssymb}
    - \\usepackage{enumitem}
    - \\usepackage{xcolor}
    - \\usepackage{mdframed}
    - |
      \\definecolor{light-gray}{gray}{0.95}
    - |
      \\newenvironment{algorithmbox}
        {\\begin{mdframed}[backgroundcolor=light-gray,hidealllines=true]\\begin{flushleft}}
        {\\end{flushleft}\\end{mdframed}}
---

"""


def clean_and_format_markdown(content):
    """
    Clean and format markdown content with special handling for code blocks containing math.
    """
    lines = content.split('\n')
    formatted_lines = []
    in_list = False
    list_indent = 0
    in_code_block = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Check if we're in a list item
        if line.lstrip().startswith(('- ', '* ', '+ ', '1. ')):
            in_list = True
            list_indent = len(line) - len(line.lstrip())
            formatted_lines.append(line)

            # Look ahead for code block
            if i + 1 < len(lines) and '```' in lines[i + 1]:
                # Check if next lines contain math symbols
                has_math = False
                temp_i = i + 2
                while temp_i < len(lines) and '```' not in lines[temp_i]:
                    if '$' in lines[temp_i]:
                        has_math = True
                        break
                    temp_i += 1

                if has_math:
                    # Handle code block with math using custom environment
                    formatted_lines.append(' ' * (list_indent + 2) + '\\begin{algorithmbox}')
                    i += 2  # Skip the ``` line

                    while i < len(lines) and '```' not in lines[i]:
                        if lines[i].strip():
                            # Keep the math expressions intact
                            formatted_lines.append(' ' * (list_indent + 4) + lines[i].strip() + ' \\\\')
                        i += 1

                    formatted_lines.append(' ' * (list_indent + 2) + '\\end{algorithmbox}')
                    formatted_lines.append('')
                else:
                    # Handle regular code block
                    formatted_lines.append('```')
                    i += 1
                    while i < len(lines) and '```' not in lines[i]:
                        if lines[i].strip():
                            formatted_lines.append(lines[i])
                        i += 1
                    formatted_lines.append('```')

        else:
            formatted_lines.append(line)
            if not line.strip():
                in_list = False
        i += 1

    return '\n'.join(formatted_lines)


def combine_markdown_files(input_dir, output_file):
    """
    Enhanced version of combine_markdown_files function with proper error handling
    """
    try:
        subprocess.run(['pandoc', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pandoc is not installed. Please install pandoc first.")
        return

    md_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.md')])

    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
        # Write enhanced YAML header
        temp_file.write(update_yaml_header())

        # Process each markdown file
        for i, md_file in enumerate(md_files, 1):
            print(f"Processing {md_file}...")

            try:
                content = read_file_safely(os.path.join(input_dir, md_file))
                title = extract_title(content)

                if title:
                    temp_file.write(f"\\chapter{{{title}}}\n\n")
                    content = clean_content(content)

                # Clean and format the content
                formatted_content = clean_and_format_markdown(content)
                temp_file.write(formatted_content)
                temp_file.write("\n\n")

            except Exception as e:
                print(f"Error processing file {md_file}: {str(e)}")
                continue

    # Update pandoc command with enhanced options
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
            '--from',
            'markdown+raw_tex+tex_math_dollars+tex_math_single_backslash+lists_without_preceding_blankline+fenced_code_blocks+fenced_code_attributes',
            '--wrap=preserve'
        ], check=True, capture_output=True, text=True)
        print(f"Successfully created PDF: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting to PDF: {e}")
        print(f"Pandoc stderr output: {e.stderr}")
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