import re
import nltk
from bs4 import BeautifulSoup
import contractions

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# NLTK resource guard (safe, runs once per environment)
# =========================================================
def _ensure_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet"
    }

    for res, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


_ensure_nltk_resources()


# =========================================================
# Shared resources (initialized once)
# =========================================================
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def normalize_category(col):
    return (
        col.astype(str)
           .str.strip()                      # remove leading/trailing spaces
           .str.replace(r"\s+", " ", regex=True)   # collapse multiple spaces
           .str.replace(r"\s*&\s*", " & ", regex=True)  # normalize & spacing
           .str.lower()                      # unify casing
    )
# =========================================================
# Normalization helpers
# =========================================================
def remove_html(text: str) -> str:
    """Remove HTML/XML tags."""
    return BeautifulSoup(text, "html.parser").get_text()


def normalize_uncertainty(text: str) -> str:
    """
    Normalize design uncertainty markers.
    '?' is replaced by a semantic token.
    """
    return re.sub(r"\?", " UNCERTAIN ", text)


def normalize_function_calls(text: str) -> str:
    """
    Normalize function calls:
    f() -> f
    setState() -> setState
    """
    return re.sub(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*\)",
        r"\1",
        text
    )


# =========================================================
# Main preprocessing function
# =========================================================
def preprocess_comment(text):
    """
    Preprocess SATD comment for classical ML models (e.g., XGBoost).

    Steps:
    - HTML removal
    - Contraction expansion
    - Uncertainty normalization
    - Function-call normalization
    - Tokenization
    - Alphanumeric + underscore filtering
    - Lowercasing
    - Stopword removal
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # Clean & normalize text
    text = remove_html(text)
    text = contractions.fix(text)
    text = normalize_uncertainty(text)
    #text = normalize_function_calls(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Keep meaningful tokens: words, numbers, identifiers
    tokens = [
        t.lower()
        for t in tokens
        if re.fullmatch(r"[a-zA-Z0-9_]+", t)
    ]

    # Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS]

    # Lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)
