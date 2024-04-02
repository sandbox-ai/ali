import json
import numpy as np
from copy import deepcopy
import re
import os


__author__ = "SandboxAI Team"
__copyright__ = "Copyright 2023, Team Research"
__credits__ = ["SandboxAI"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "SanboxAI Team"
__email__ = "sandboxai <dot> org <at> proton <dot> me"
__status__ = "Development"


def basic_split(text: str, max_chunk_size: int, min_chunk_size: int, metadata: dict = None):
    """Chunk a list into smaller lists of a given size, sliding in steps."""
    # Initialize variables:
    chunks = []
    
    if metadata is None:
        metadata = {}
    
    count = 0
    
    # Slide over the text:
    for idx in range(0, len(text), max_chunk_size):
        # Get a chunk of max_chunk_size and append to return list:
        chunk_text = text[idx : idx+max_chunk_size]

        # Append index to metadata:
        chunk_metadata = metadata.copy()
        chunk_metadata['idx'] = count

        # Create a chunk as a dictionary with text and metadata:
        chunk = {
            "text": chunk_text,
            "metadata": chunk_metadata
        }
        chunks.append(chunk)

        # Update chunk count:
        count += 1

    # If the last chunk is too short, remove it:
    if len(chunks[-1]["text"]) < min_chunk_size:
        chunks.pop(-1)
    
    return chunks


def window_slide_split(text: str, max_chunk_size: int, min_chunk_size: int, sliding_step: int, metadata: dict = None):
    """Chunk a list into smaller lists of a given size, sliding in steps."""
    # Initialize variables:
    chunks = []
    
    if metadata is None:
        metadata = {}
    
    # Slide over the text:
    for idx in range(0, len(text), sliding_step):
        # Get a chunk of max_chunk_size and append to return list:
        chunk_text = text[idx : idx+max_chunk_size]

        # Append index to metadata:
        chunk_metadata = metadata.copy()
        chunk_metadata['start_idx'] = idx
        chunk_metadata['end_idx'] = min(idx + max_chunk_size, len(text))

        # Create a chunk as a dictionary with text and metadata:
        chunk = {
            "text": chunk_text,
            "metadata": chunk_metadata
        }
        chunks.append(chunk)

    # If the last chunk is too short, remove it:
    if len(chunks[-1]["text"]) < min_chunk_size:
        chunks.pop(-1)
    
    return chunks


def window_slide_split_paragraphs(text: str, max_chunk_size: int, min_chunk_size: int, sliding_step: int, metadata: dict = None):
    """Chunk a text into smaller lists based on paragraph sizes and sliding window approach, with optional inherited metadata."""
    # Split the text into paragraphs
    paragraphs = text.split('\n')

    # Initialize variables
    merged_paragraphs = []
    temp_paragraph = ''
    temp_start_idx = 0
    chunks = []

    # Default metadata if not provided
    if metadata is None:
        metadata = {}

    # Improved merging logic with index tracking
    for paragraph in paragraphs:
        if not temp_paragraph:  # If starting a new merge, record the start index
            temp_start_idx = text.find(paragraph, temp_start_idx)

        temp_paragraph += paragraph + '\n'

        if len(temp_paragraph) >= min_chunk_size:
            end_idx = temp_start_idx + len(temp_paragraph)
            merged_paragraphs.append((temp_paragraph, temp_start_idx, end_idx))
            temp_paragraph = ''
            temp_start_idx = end_idx  # Update start index for next paragraph

    # Add the last temp paragraph if it exists
    if temp_paragraph:
        end_idx = temp_start_idx + len(temp_paragraph)
        merged_paragraphs.append((temp_paragraph, temp_start_idx, end_idx))

    # Process each merged paragraph
    for paragraph, start_idx, end_idx in merged_paragraphs:
        if len(paragraph) > max_chunk_size:
            # Apply sliding window approach for long paragraphs
            for idx in range(0, len(paragraph), sliding_step):
                chunk_text = paragraph[idx : idx+max_chunk_size]
                # Check if the chunk is shorter than min_chunk_size
                if len(chunk_text) >= min_chunk_size or idx == 0:  # Always include the first chunk
                    # Extract laws from chunk:
                    laws = extract_laws(chunk_text)

                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'laws': laws,
                        'start_idx': start_idx + idx,
                        'end_idx': min(start_idx + idx + max_chunk_size, end_idx)
                    })
                    chunk = {"text": chunk_text, "metadata": chunk_metadata}
                    chunks.append(chunk)
        else:
            # Extract laws from chunk:
            laws = extract_laws(paragraph)

            # Treat as a single chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'laws': laws,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
            chunk = {"text": paragraph, "metadata": chunk_metadata}
            chunks.append(chunk)

    return chunks


def extract_text_with_path(source: dict, parent_path=''):
    """
    Extracts text and its path from a nested JSON structure.
    """
    extracted = []
    for key, value in source.items():
        current_path = f"{parent_path} - {key}" if parent_path else key
        if isinstance(value, dict):
            # Recursive call for nested dictionaries
            extracted.extend(extract_text_with_path(value, current_path))
        else:
            extracted.append({'text': value, 'metadata': {'location': current_path}})
    return extracted


def extract_articles(data: dict, metadata: dict = None):
    """
    Extracts articles from the given JSON structure and returns a list of dictionaries
    with article text and metadata.

    :param data: Dictionary representing the JSON structure.
    :return: List of dictionaries, each representing an article and its metadata.
    """
    def navigate_contents(titulo, title, contents, capitulo=None, capitulo_title=None):
        """
        Navigates the contents of a title or chapter to find articles.

        :param titulo: The title under which articles or chapters are nested.
        :param title: The actual title text.
        :param contents: The contents of the title or chapter.
        :param capitulo: The chapter under which articles are nested, if any.
        :param capitulo_title: The actual chapter title text, if any.
        :return: None; results are appended to the articles list.
        """
        for key, value in contents.items():
            if key.startswith("Articulo"):
                article_metadata = metadata.copy()

                # Append article details to the list
                article_metadata['titulo'] = f"{titulo} - {title}"
                if capitulo: article_metadata['capitulo'] = f"{capitulo} - {capitulo_title}"
                article_metadata['articulo'] = f"{key}"

                lengths.append(len(value))
                articles.append({'text': value, 'metadata': article_metadata})
            elif key.startswith("Capitulo"):
                # Navigate into the chapter
                navigate_contents(titulo, title, value['contents'], key, value['title'])

    if metadata is None:
        metadata = {}

    lengths = []
    articles = []
    for titulo, title_data in data.items():
        navigate_contents(titulo, title_data['title'], title_data['contents'])

    print(f"75% of Articles are below {np.percentile(lengths, 75)} chars.")
    return articles


def extract_laws(text):
    # Regular expression pattern to match the law mentions
    pattern_1 = r"Ley N° \d+\.\d+"
    pattern_2 = r"Ley N° \d+/\d+"

    # Find all instances in the text
    law_mentions_1 = re.findall(pattern_1, text)
    law_mentions_2 = re.findall(pattern_2, text)

    # Format the instances to remove 'N°' and get unique values
    formatted_laws = set(law.replace("Ley N° ", "") for law in law_mentions_1 + law_mentions_2)
    formatted_laws = set(law.replace(".", "") for law in formatted_laws)

    # Convert the set back to a list for return
    return list(formatted_laws)


if __name__ == "__main__":
    # Initialize chunks list:
    chunks = []

    # Define paths:
    source_txt_path = './data/ALI-raw/decreto.json'
    target_chunks_path = 'data/ALI/decreto_chunks.json'

    # Define chunking parameters:
    max_chunk_size = 680
    sliding_chunk_size = 200
    min_chunk_size = 500

    # Load text data:
    if os.path.exists(source_txt_path):
        with open(source_txt_path, 'r', encoding='utf-8') as file:
            #source = file.read()
            source = json.load(file)

    # Split by pattern:
    segments = extract_articles(source, metadata={'documento': 'Decreto', 'fecha': '20-12-2023'})

    # Split segments into chunks:
    for segment in segments:
        chunks += window_slide_split_paragraphs(segment['text'],
                                                max_chunk_size,
                                                min_chunk_size,
                                                sliding_chunk_size,
                                                segment['metadata'])


    """chunks += window_slide_split_paragraphs(source,
                                            max_chunk_size,
                                            min_chunk_size,
                                            sliding_chunk_size,
                                            {'documento': 'Consideraciones', 'fecha': '20-12-2023'})"""

    # Save to JSON file:
    with open(target_chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
