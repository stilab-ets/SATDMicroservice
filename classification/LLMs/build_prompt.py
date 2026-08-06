import os
from pathlib import Path

def load_prompt_template(filename: str) -> str:
    """Load a prompt template from a text file."""

    #template_dir = os.path.join(os.path.dirname(__file__))
    #filepath = os.path.join(template_dir, filename)

    BASE_DIR = Path(__file__).resolve().parent
    filepath = BASE_DIR / "prompts" / filename

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_categories() -> list:
    """Load the list of SATD categories."""
    categories_path = Path(__file__).resolve().parent / "prompts" / "categories.txt"
    with open(categories_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
def build_zero_shot_prompt(comment: str, context: dict = None) -> str:
    """
    Build a zero-shot prompt for SATD classification.

    Args:
        comment : The SATD comment to classify
        context : Optional dict with keys:
                    - 'file_path'      : path/to/File.java
                    - 'language'       : Java, Python, Go, etc.
                    - 'context'         : surrounding code snippet OR summary of the contex
    """
    wrapped_comment = f"```{comment}```"
    
    if context:
        # ======================================================
        # Build structured context block
        # ======================================================
        
        file_path       = context.get("file_path",       "unknown")
        language        = context.get("language",        "unknown")
        context_content = context.get("context", "")

        wrapped_context = (
            f"File     : {file_path}\n"
            f"Language : {language}\n"
            f"Context  :\n```{context_content}\n```"
        )
        #template = load_prompt_template("zero_test.txt")

        template = load_prompt_template("prompt_Zero_shot_with_context.txt")
        return template.format(
            comment=wrapped_comment,
            context=wrapped_context
        )
    else:
        template = load_prompt_template("prompt_Zero_shot_without_context.txt")
        return template.format(
            comment=wrapped_comment   
            )
    
def format_examples(few_shot_examples: list) -> str:
    """Format few-shot examples into a readable string."""
    examples_text = ""
    for i, ex in enumerate(few_shot_examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Comment: {ex['comment']}\n"
        examples_text += f"Label: {ex['label']}\n\n"
    return examples_text.strip()

def preprocess_context(context: str) -> str:
    """
    Return context as-is — no preprocessing.
    What is in the dataset is exactly what goes into the prompt.
    """
    if not context or not str(context).strip():
        return ""
    
    return str(context).strip()

def build_few_shot_prompt(comment: str, few_shot_examples: list, context: dict = None) -> str:
    """
    Build a few-shot prompt for SATD classification.

    Args:
        comment          : The SATD comment to classify
        few_shot_examples: List of few-shot examples
        context          : Optional dict with keys:
                             - 'file_path'     : path/to/File.java
                             - 'language'      : Java, Python, Go, etc.
                             - 'context'         : surrounding code snippet OR summary of the context
    """

    examples_text   = format_examples(few_shot_examples)
    wrapped_comment = f"```{comment}```"

    if context:
        # ======================================================
        # Build structured context block
        # ======================================================
        surrounded_code = preprocess_context(context.get("surrounded_code", ""))

        file_path      = context.get("file_path",      "unknown")
        language       = context.get("language",       "unknown")
        context_content = context.get("context", "")

        wrapped_context = (
            f"File     : {file_path}\n"
            f"Language : {language}\n"
            f"Context  :\n```{context_content}\n```"
        )

        template = load_prompt_template("prompt_few_shot_with_context.txt")
        return template.format(
            examples_text=examples_text,
            comment=wrapped_comment,
            context=wrapped_context
        )
    else:
        template = load_prompt_template("prompt_few_shot_without_context.txt")
        return template.format(
            examples_text=examples_text,
            comment=wrapped_comment
        )

