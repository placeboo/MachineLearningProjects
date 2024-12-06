"""
In this file, I will leverage LLM to generate notes for the lecture transcript.
"""

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import PromptTemplate
from dotenv import load_dotenv
import os


SYSTEM_PROMPT = PromptTemplate(
    template="""
You are a graduate teaching assistant in computer science with extensive knowledge of machine learning theory and applications. Your task is to analyze the provided lecture transcript section and create comprehensive, graduate-level lecture notes. Your notes should reflect deep technical understanding and be suitable for PhD-level machine learning exam preparation.

Generate detailed lecture notes following this structure:

TITLE: [Topic Name]

1. THEORETICAL FOUNDATIONS (In-depth coverage)
   - Core mathematical principles and frameworks
   - Formal definitions with precise mathematical notation
   - Fundamental theorems and their implications
   - Derivations of key equations and proofs
   - Theoretical constraints and assumptions


2. KEY CONCEPTS AND METHODOLOGY
   A. Essential Concepts
      - Detailed explanation of each core concept
      - Mathematical formulation and notation
      - Relationships between concepts
      - Edge cases and special conditions
   
   B. Algorithms and Methods
      - Step-by-step algorithmic descriptions
      - Pseudocode for key algorithms
      - Complexity analysis (time and space)
      - Convergence properties and proofs
      - Optimization techniques and variations

3. APPLICATIONS AND CASE STUDIES
   - Example mentioned in the lecture (if applicable)
   - Implementation variations for different scenarios
   - Performance comparisons
   - Limitations and considerations in practice

4. KEY TAKEAWAYS AND EXAM FOCUS
   - Essential theoretical results
   - Critical implementation details
   - Common exam questions and approaches
   - Important proofs and derivations to remember
   - Key equations and their interpretations

Format Requirements:
- Use $...$ for ALL mathematical expressions
- Bold for important terms and definitions
- Numbered lists for sequential processes
- Bullet points for related items
- Include clear diagram descriptions where relevant
- Use tables for comparative analysis
- Include complexity notation (e.g., $O(n)$) where applicable

IMPORTANT:
1. Maintain rigorous mathematical precision
2. Include concrete examples for complex concepts
3. Provide intuitive explanations alongside formal definitions
4. Connect to broader machine learning theory
5. Address both theoretical and practical aspects
6. Include recent developments and research directions
7. Highlight exam-relevant material

Please ensure you fully explain EVERY concept mentioned, providing both intuitive understanding and formal mathematical treatment. Do not skip intermediate steps in derivations or proofs.

Transcript: {transcript}

Summary:
    """
)

load_dotenv()
llm = AzureOpenAI(
        model=os.getenv('MODEL_NAME'),
        engine=os.getenv('ENGINE'),
        temperature=0.7,
        max_tokens=5000,
        retry=2,
        timeout=60
    )

def generate_summary(prompt: PromptTemplate,
                     file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        transcript = file.read()

    formatted_prompt = prompt.format(transcript=transcript)
    response = llm.complete(formatted_prompt)
    return response.text

def generate_all_summary(input_dir: str,
                         output_dir: str,
                         prompt: PromptTemplate) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            print(f"Generating summary for {file_name}")
            file_path = os.path.join(input_dir, file_name)
            summary = generate_summary(prompt, file_path)
            print()

            # save the summary to md file
            md_filename = file_name.replace(".txt", ".md")
            with open(os.path.join(output_dir, md_filename), 'w', encoding='utf-8') as file:
                file.write(summary)
            print(f"Generated summary for {file_name}")

def main():
    input_dir = 'lecture_scripts'
    output_dir = 'lecture_notes'

    generate_all_summary(input_dir, output_dir, SYSTEM_PROMPT)

if __name__ == "__main__":
    main()