import os
import gc
import torch
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm.auto import tqdm
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .manual_batch import _batch_files
from .voxprofile.src.model.age_sex.wavlm_demographics import WavLMWrapper
from .voxprofile.src.model.age_sex.whisper_demographics import WhisperWrapper

@dataclass
class TransformersAgeSexModel:
    backend: str
    model: Any
    agesex_model_name: str
    device: str
    cache_dir: Optional[str]
    model_batch_size: int
    compute_type: str

SEX_UNIQUE_LABELS = ["Female", "Male"]

def load_age_sex_model(
    agesex_model_name: str = "tiantiaf/wavlm-large-age-sex",
    device: str = "auto",
    cache_dir: Optional[str] = None,
    model_batch_size: int = 16,
    backend: str = "auto",
    compute_type: Optional[str] = None,
) -> TransformersAgeSexModel:
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
        
        if compute_type == "float16" and device == "cuda":
            model = model.half()

        return TransformersAgeSexModel(
            backend=backend,
            model=model,
            agesex_model_name=agesex_model_name,
            device=device,
            cache_dir=cache_dir,
            model_batch_size=model_batch_size,
            compute_type=compute_type,
        )
    
    raise ValueError(f"Unsupported model or backend: {agesex_model_name}")
def _wavlm_predict_batch_inference(
    batch_segments: List[Dict[str, Any]], 
    model_wrapper: Any
) -> List[Dict[str, Any]]:
    """Loads audio, tracks actual sample lengths, pads them dynamically, and runs batch inference."""
    device = model_wrapper.device
    model = model_wrapper.model
    
    audio_tensors = []
    lengths = []
    max_len = 0
    
    # 1. Load raw audio and track original lengths (From your working snippet)
    for seg in batch_segments:
        audio, sr = sf.read(seg["seg_filename"])
        if audio.ndim > 1:
            audio = audio[:, 0]
            
        tensor = torch.tensor(audio, dtype=torch.float32)
        audio_tensors.append(tensor)
        lengths.append(len(tensor))
        if len(tensor) > max_len:
            max_len = len(tensor)
            
    # 2. Dynamic zero-padding to make shapes uniform for stacking
    padded_tensors = []
    for tensor in audio_tensors:
        pad_size = max_len - len(tensor)
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size), "constant", 0.0)
        padded_tensors.append(tensor)
        
    # 3. Stack arrays and lengths into shapes expected by the model
    input_batch = torch.stack(padded_tensors).to(device)
    input_lengths = torch.tensor(lengths, dtype=torch.long).to(device)
    
    if getattr(model_wrapper, "compute_type", None) == "float16" and device == "cuda":
        input_batch = input_batch.half()
        
    # 4. Model Inference Execution with lengths
    with torch.no_grad():
        # Passing length=input_lengths directly to fix the masking IndexError!
        wavlm_age_outputs, wavlm_sex_outputs = model(input_batch, length=input_lengths)
        
        # Age extraction
        age_preds = (wavlm_age_outputs.detach().cpu().numpy() * 100).flatten()
        
        # Sex probability extraction
        sex_probs = F.softmax(wavlm_sex_outputs, dim=1)
        sex_indices = torch.argmax(sex_probs, dim=1).detach().cpu().tolist()
        
    # 5. Format structured outputs (Note: Ensure SEX_UNIQUE_LABELS is defined globally)
    batch_results = []
    for idx, sex_idx in enumerate(sex_indices):
        batch_results.append({
            "age": float(age_preds[idx]),
            "sex": SEX_UNIQUE_LABELS[sex_idx]  # Ensure this array matches your setup
        })
        
    return batch_results



def _predict_demographics_batch(
    batch: List[Dict[str, Any]],
    model: TransformersAgeSexModel,
    cache: bool = False,
) -> List[Dict[str, Any]]:
    """Manages cache validation and coordinates batch routing blocks."""
    files_to_predict = []
    file_indices = []
    results = [None] * len(batch)
    for i, seg in enumerate(batch):
        demo_cache = seg['seg_filename'].replace(".wav", "_demographics.txt")

        if cache and os.path.exists(demo_cache):
            try:
                with open(demo_cache, "r", encoding="utf-8") as cache_file:
                    cached_text = cache_file.read().strip()
                if not cached_text.startswith("[AGE_SEX_PREDICTION_FAILED:"):
                    # Cache format pattern: "Age: 32.5 | Sex: Male"
                    parts = cached_text.split(" | ")
                    age = float(parts[0].split(": ")[1])
                    sex = parts[1].split(": ")[1]
                    results[i] = {"age": age, "sex": sex}
                    continue
            except Exception:
                pass # Stale cache configuration fallback
                
        files_to_predict.append(seg)
        file_indices.append(i)

    # Execute processing if active inputs exist
    if files_to_predict:
        batch_outputs = _wavlm_predict_batch_inference(files_to_predict, model)
        
        for batch_idx, output in zip(file_indices, batch_outputs):
            results[batch_idx] = output
            
            if cache:
                cache_path = batch[batch_idx]["demo_cache"]
                with open(cache_path, "w", encoding="utf-8") as cache_file:
                    cache_file.write(f"Age: {output['age']:.1f} | Sex: {output['sex']}")

    return results

def predict_demographics_segments(
    model: Any,
    segments: pd.DataFrame,
    output_dir: str,
    cache: bool = True,
    batch_size: Optional[float] = 30.0,  # Bumped from 1.0 to 30.0 to allow actual batching
    min_duration_samples: int = 1600,
) -> Dict[str, Any]:
    """Primary pipeline executor to slice data structures and append metrics."""
    batches = _batch_files(segments, output_dir, batch_size)

    predictions_map = {}
    for batch in tqdm(batches, desc=f"Processing {len(batches)} demographic batches"):
        batch_results = _predict_demographics_batch(batch, model, cache=cache)
        
        for seg, res in zip(batch, batch_results):
            predictions_map[seg["idx"]] = res

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return predictions_map

