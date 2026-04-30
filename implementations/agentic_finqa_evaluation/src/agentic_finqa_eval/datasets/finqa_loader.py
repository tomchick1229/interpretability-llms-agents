import re
import os
import json
from pathlib import Path
from typing import List, Optional

def load_finqa(n: Optional[int] = None,
    finqa_dir: str = "data/finqa",
    finqa_filename: str = "dev_updated_with_answer.json",
) -> List[dict]:
    """
    Simple loader for FinQA datasets.
    Parameters
    n : int, optional
        Maximum number of samples to return.
    finqa_dir : str, optional
        Directory containing the FinQA dataset.
    Returns
    list of dict
    """
    f = open(os.path.join(finqa_dir, finqa_filename))
    samples = json.load(f)
    if n is not None:
        samples = samples[:n]

    return samples