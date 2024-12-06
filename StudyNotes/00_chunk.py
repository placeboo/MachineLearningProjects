"""
In this file, I will break the lecture transcript into chunks and save them in the respective files.
The transcript format
CHAPTER 1: INTRODUCTION
1.1 Decision Trees
1.1.1 xxxx
...
1.2 xxx
1.2.1 xxxx
...
...
CHAPTER 2: xxxx

Each file is at x.x level
"""

import re
import os

import re
import os
from pathlib import Path


def process_transcript(input_file):
    """
    Process lecture transcript and split it into separate files based on section numbers.

    Args:
        input_file (str): Path to the input transcript file
    """
    # Create output directory if it doesn't exist
    output_dir = Path("lecture_scripts")
    output_dir.mkdir(exist_ok=True)

    # Regular expressions for matching chapter and section headers
    chapter_pattern = r'^CHAPTER\s+\d+\s+.*$'
    section_pattern = r'^\d+\.\d+\s+.*$'
    subsection_pattern = r'^\d+\.\d+\.\d+\s+.*$'

    current_chapter = ""
    current_section = ""
    current_content = []

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()

            # Skip empty lines at the beginning of sections
            if not line and not current_content:
                continue

            # Check if line is a chapter header
            if re.match(chapter_pattern, line):
                # Save previous section if exists
                if current_section and current_content:
                    save_section(output_dir, current_section, current_content)
                current_chapter = line
                current_content = []

            # Check if line is a section header
            elif re.match(section_pattern, line) and not re.match(subsection_pattern, line):
                # Save previous section if exists
                if current_section and current_content:
                    save_section(output_dir, current_section, current_content)
                current_section = line
                current_content = [current_chapter, "", current_section, ""]  # Add chapter header and blank line

            # Add content to current section
            else:
                if current_section:
                    current_content.append(line)

        # Save last section
        if current_section and current_content:
            save_section(output_dir, current_section, current_content)

        print(f"Successfully processed transcript into {output_dir}")

    except Exception as e:
        print(f"Error processing file: {str(e)}")


def save_section(output_dir, section_header, content):
    """
    Save section content to a file.

    Args:
        output_dir (Path): Directory to save files
        section_header (str): Section header (used for filename)
        content (list): Lines of content to save
    """
    # Extract section number for filename
    section_num = re.match(r'(\d+\.\d+)', section_header).group(1)
    filename = f"section_{section_num.replace('.', '_')}.txt"

    with open(output_dir / filename, 'w', encoding='utf-8') as file:
        file.write('\n'.join(content))


def main():
    # You can modify this path to match your transcript file location
    input_file = "CS7641 ML Lectures All Chapters Raw.txt"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    process_transcript(input_file)


if __name__ == "__main__":
    main()


