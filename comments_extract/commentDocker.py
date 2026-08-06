import os
from typing import List, Iterable, Optional,Any, Dict, List, Union
CommentItem = Dict[str, Union[int, str]]
from pathlib import Path

class commentExtractorDocker:
    def __init__(self, file_path=None):
        self._file_path = file_path

    def extract_comments(self, source_code: str):
        results = []
        current_block_lines = []
        block_start_line = None
        line_number = 0  # stays 0 if source_code is empty

        for line_number, line in enumerate(source_code.splitlines(), 1):
            is_comment = False
            comment_start_index = line.find('#')

            if comment_start_index != -1:
                # Same heuristic: '#' counts only if not inside quotes
                before_hash = line[:comment_start_index]
                if before_hash.count("'") % 2 == 0 and before_hash.count('"') % 2 == 0:
                    is_comment = True
                    comment_text = line[comment_start_index:].strip()
                    current_block_lines.append(comment_text)

                    if block_start_line is None:
                        block_start_line = line_number

            # If this line isn't a comment and a block is open, flush the block
            if not is_comment and current_block_lines:
                results.append({
                    'comment': "\n".join(current_block_lines),
                    'start_line': block_start_line,
                    'end_line': line_number - 1
                })
                current_block_lines = []
                block_start_line = None

        # Handle file ending with a comment block
        if current_block_lines:
            results.append({
                'comment': "\n".join(current_block_lines),
                'start_line': block_start_line,
                'end_line': line_number
            })

        return results

# --- Example Usage ---

if __name__ == "__main__":
    # Create a dummy file with both grouped and inline comments
    file_content = """
# This is the first line
# of a grouped comment block.
version: '3.8'

# This is a single line comment.
services:
  web:
    build: .
    ports:
      - "5000:5000" # This is a valid inline comment
    environment:
      PASSWORD: "my-secure-pass#word" # The hash here is part of the string
"""
    
    
    #with open(file_path, "w", encoding='utf-8') as f:
    #    f.write(file_content)
    # root = parent of folder D (i.e., the common root)
    ROOT = Path(__file__).resolve().parents[1]
    file_path = ROOT / "tmp" / "DockerFile"
    
    # Create an instance of the class and extract comments
    extractor = commentExtractorDocker()
    extracted_comments = extractor.extract_comments(file_content)

    print(extracted_comments) 
    
  