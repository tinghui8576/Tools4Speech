import os
import gc
import torch
import pandas as pd
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
from .char_inference import _batch_files, _char_predict_batch_inference, TransformersCharModel
from .voxprofile.src.model.age_sex.wavlm_demographics import WavLMWrapper
from .voxprofile.src.model.age_sex.whisper_demographics import WhisperWrapper


SEX_UNIQUE_LABELS = ["Female", "Male"]

def load_age_sex_model(
    agesex_model_name: str = "tiantiaf/wavlm-large-age-sex",
    device: str = "auto",
    cache_dir: Optional[str] = None,
    model_batch_size: int = 16,
    backend: str = "auto",
    compute_type: Optional[str] = None,
) ->TransformersCharModel:
    """Initialise and return a demographic prediction model via Voxprofile."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "float32"

    if "wavlm" in agesex_model_name:
        backend = "wavlm-large"
        model = WavLMWrapper.from_pretrained(agesex_model_name).to(device)
        model.eval() 
    elif "whisper" in agesex_model_name:
        backend = 'whisper'
        raise ValueError(f"Unsupported model: {agesex_model_name}")
    else:
        raise ValueError(f"Unsupported backend: {agesex_model_name}")
    
    if compute_type == "float16" and device == "cuda":
        model = model.half()

    return TransformersCharModel(
        backend=backend,
        model=model,
        Char_model_name=agesex_model_name,
        device=device,
        cache_dir=cache_dir,
        model_batch_size=model_batch_size,
        compute_type=compute_type,
    )
    
def predict_demographics_segments(
    model: Any,
    segments: pd.DataFrame,
    output_dir: str,
    cache: bool = True,
    batch_size: Optional[float] = 30.0,
    min_duration_samples: int = 1600,
) -> Dict[str, Any]:
    """
    Slices segments into dynamic batches, verifies disk-cached files, 
    and passes uncached elements to WavLM batch inference before returning results.
    """
    batches = _batch_files(segments, output_dir, batch_size, max_duration_samples= 15.0)
    predictions_map = {}

    for batch in batches:
        files_to_predict = []
        file_indices = []
        batch_results = [None] * len(batch)

        for i, seg in enumerate(batch):
            demo_cache = seg['seg_filename'].replace(".wav", "_demographics.txt")

            if cache and os.path.exists(demo_cache):
                try:
                    with open(demo_cache, "r", encoding="utf-8") as cache_file:
                        cached_text = cache_file.read().strip()
                    if not cached_text.startswith("[AGE_SEX_PREDICTION_FAILED:"):
                        parts = cached_text.split(" | ")
                        age = float(parts[0].split(": ")[1])
                        sex = parts[1].split(": ")[1]
                        batch_results[i] = {"age": age, "sex": sex}
                        continue
                except Exception:
                    pass  

            files_to_predict.append(seg)
            file_indices.append(i)

        # Model Inference execution block 
        if files_to_predict:
            outputs = _char_predict_batch_inference(files_to_predict, model, max_duration_samples=15.0)

            # Safely detach and process the outputs
            age_preds = (outputs[0].detach().cpu().numpy() * 100).flatten()
            sex_probs = F.softmax(outputs[1], dim=1)
            sex_indices = torch.argmax(sex_probs, dim=1).detach().cpu().tolist()
            # Map generated data targets back into batch metrics
            for batch_idx, sex_idx  in zip(file_indices, sex_indices):
                current_age = float(age_preds[batch_idx])
                current_sex = SEX_UNIQUE_LABELS[sex_idx]

                batch_results[batch_idx] = {
                    "age": current_age,
                    "sex": current_sex
                }

                if cache:
                    # Construct cache file path smoothly
                    cache_path = batch[batch_idx].get("demo_cache", batch[batch_idx]['seg_filename'].replace(".wav", "_demographics.txt"))
                    with open(cache_path, "w", encoding="utf-8") as cache_file:
                        cache_file.write(f"Age: {current_age:.1f} | Sex: {current_sex}")

        # Save results back using their true global identifiers
        for seg, res in zip(batch, batch_results):
            predictions_map[seg["idx"]] = res

        # Clean up system memory after every batch iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return predictions_map