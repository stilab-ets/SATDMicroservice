import sentence_transformers
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # suppress symlink warning
os.environ["HF_TOKEN"] = "key"         # suppress unauthenticated warning

# Load once globally
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_the_most_relevant_items_for_an_item(query_embed, keys_embed, k, cos_sim=None):
    """
    Return top-k (index, similarity) pairs.
    If cos_sim is given, use it directly. Otherwise compute cosine similarity.
    """
    if cos_sim is None:
        cos_sim = sentence_transformers.util.cos_sim(query_embed, keys_embed)

    similarities = cos_sim.squeeze().tolist()

    if isinstance(similarities, float):
        similarities = [similarities]

    itemId_similarity = dict(zip(range(len(keys_embed)), similarities))
    itemId_similarity = dict(
        sorted(itemId_similarity.items(), key=lambda item: item[1], reverse=True)
    )
    itemId_similarity = [(idx, itemId_similarity[idx]) for idx in list(itemId_similarity)[:k]]
    return itemId_similarity


def sample_dynamic_few_shot(query_comment, X_text, y, train_idx, id_to_label, k=3):
    """
    Dynamic few-shot retrieval using MiniLM sentence embeddings.
    """
    train_comments = X_text[train_idx].tolist()

    if len(train_comments) == 0:
        return []

    query_embed = torch.tensor(embedder.encode([query_comment], convert_to_numpy=True))
    keys_embed = torch.tensor(embedder.encode(train_comments, convert_to_numpy=True))

    top_k = get_the_most_relevant_items_for_an_item(
        query_embed,
        keys_embed,
        min(k, len(train_comments))
    )

    examples = []
    for local_idx, similarity in top_k:
        global_idx = train_idx[local_idx]
        examples.append({
            "comment": X_text[global_idx],
            "label": id_to_label[y[global_idx]],
            "similarity": round(float(similarity), 4)
        })

    return examples


def sample_dynamic_few_shot_stratified(query_comment, X_text, y, train_idx, id_to_label, k=1):
    """
    Stratified dynamic few-shot retrieval using MiniLM.
    Retrieves top-k per category.
    Total prompt examples = k x number of categories.
    """
    examples = []
    query_embed = torch.tensor(embedder.encode([query_comment], convert_to_numpy=True))

    for label_id in np.unique(y[train_idx]):
        cat_mask = np.where(y[train_idx] == label_id)[0]
        cat_indices = train_idx[cat_mask]
        cat_comments = X_text[cat_indices].tolist()

        if len(cat_comments) == 0:
            continue

        keys_embed = torch.tensor(embedder.encode(cat_comments, convert_to_numpy=True))

        top_k = get_the_most_relevant_items_for_an_item(
            query_embed,
            keys_embed,
            min(k, len(cat_comments))
        )

        for local_idx, similarity in top_k:
            global_idx = cat_indices[local_idx]
            examples.append({
                "comment": X_text[global_idx],
                "label": id_to_label[label_id],
                "similarity": round(float(similarity), 4)
            })

    return examples
