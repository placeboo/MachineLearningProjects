"""
In this file, I will leverage LLM to generate notes for the lecture transcript.
"""

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import PromptTemplate
from dotenv import load_dotenv
import os


SYSTEM_PROMPT = PromptTemplate(
    template="""
    You are a graduate teaching assistant in computer science with extensive knowledge of machine learning theory and applications. Your task is to analyze the provided lecture transcript section (which contains a discussion between two professors) and create a comprehensive notes that combines the transcript content with additional graduate-level insights. The goal is to generate structured summaries that can be used for exam preparation and academic reference.
    
    Please create a structured summary that includes:
    
    0. TITLE
    
    1. OVERVIEW (2-3 sentences)
       - Main topic
    
    2. KEY CONCEPTS
       - Core ideas and fundamental principles
       - Essential definitions and terminology
       - Mathematical formulations or algorithms (if any)
       - Important relationships between concepts
       - Theoretical foundations and proofs (where applicable)
    
    3. PRACTICAL APPLICATIONS
       - Common use cases
       - Limitations and considerations
    
    4. IMPLEMENTATION DETAILS (if applicable)
       - Key steps or procedures
       - Important parameters or variables
       - Common pitfalls and how to avoid them
       - Computational complexity considerations
       - Optimization techniques and best practices
    
    5. KEY TAKEAWAYS
       - 3-5 bullet points highlighting the most exam-relevant concepts
       - Critical distinctions or comparisons with related topics
       - Common misconceptions and their clarifications
    
    Format your response using:
    - Bold text for important terms and definitions
    - Bullet points for lists and examples
    - Mathematical notation when necessary (properly formatted)
    - Short, clear explanations suitable for exam review
    
    IMPORTANT GUIDELINES:
    1. If the transcript's coverage of a topic seems insufficient for graduate-level understanding, supplement with:
       - Additional theoretical background
       - More rigorous mathematical treatments
       - Advanced algorithmic analysis
       - Recent developments and applications
       - Connections to other advanced ML concepts
    
    2. Maintain balance between:
       - Theory and practice
       - Basic concepts and advanced extensions
       - Historical foundations and current developments
       - Mathematical rigor and intuitive understanding
    
    3. Focus on graduate-level depth by including:
       - Theoretical proofs where relevant
       - Formal mathematical definitions
       - Algorithm complexity analysis
       - Implementation trade-offs
       - Research perspectives
    
    Keep the summary concise yet comprehensive, focusing on information that would be most valuable for an advanced graduate-level exam preparation. Ensure technical accuracy while making complex concepts accessible.
    
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