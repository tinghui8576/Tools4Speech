import os
import torch
import pandas as pd
import soundfile as sf
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TransformersCharModel:
    backend: str
    model: Any
    Char_model_name: str
    device: str
    cache_dir: Optional[str]
    model_batch_size: int
    compute_type: str

def _batch_files(
    segments: pd.DataFrame,
    output_dir: str,
    batch_size: Optional[float] = 30.0,
    max_duration_samples:Optional[float] = 15.0,
) ->List[List[Dict[str, Any]]]:
    os.makedirs(output_dir, exist_ok=True)
    
    segment_info = []
    for idx, row in segments.iterrows():
        seg_filename = row.get("seg_filename", f"segment_{idx}.wav")
        
        segment_info.append({
            "idx": idx,
            "speaker": row.get("speaker", "unknown"),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "seg_filename": seg_filename,
        })

    max_batch_duration = (
        float(max_duration_samples) if max_duration_samples and batch_size > 0 else float("inf")
    )

    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_duration = 0.0

    for seg in segment_info:
        seg_duration = float(seg["end_sec"] - seg["start_sec"])

        if seg_duration > max_batch_duration:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_duration = 0.0
            batches.append([seg])
            continue

        if current_batch and current_duration + seg_duration > max_batch_duration:
            batches.append(current_batch)
            current_batch = [seg]
            current_duration = seg_duration
        else:
            current_batch.append(seg)
            current_duration += seg_duration

    if current_batch:
        batches.append(current_batch)

    return batches

def _char_predict_batch_inference(
    batch_segments: List[Dict[str, Any]], 
    model_wrapper: Any,
    max_duration_samples: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Loads audio, tracks actual sample lengths, pads them dynamically, and runs batch inference."""
    device = model_wrapper.device
    model = model_wrapper.model
    
    audio_tensors = []
    lengths = []
    max_len = 0
    
    # 1. Load raw audio and track original lengths
    for seg in batch_segments:
        audio, sr = sf.read(seg["seg_filename"])
        if audio.ndim > 1:
            audio = audio[:, 0]
        if max_duration_samples and len(audio)/sr > max_duration_samples:
            audio = audio[:int(max_duration_samples*sr)]
            
        # Create standard float32 tensor on CPU first
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
        
    # 3. Stack arrays and lengths, sending them directly to the target device
    input_batch = torch.stack(padded_tensors).to(device)
    input_lengths = torch.tensor(lengths, dtype=torch.long).to(device)
        
    # 4. Model Inference Execution using Autocast
    # Determine if we should use mixed precision based on your wrapper config
    use_fp16 = getattr(model_wrapper, "compute_type", None) == "float16" and "cuda" in str(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", enabled=use_fp16):
            outputs = model(input_batch, length=input_lengths)
            
    return outputs