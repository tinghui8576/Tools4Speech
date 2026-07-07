import os
import gc
import torch
import pandas as pd
from tqdm.auto import tqdm
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
from .char_inference import _batch_files, _char_predict_batch_inference, TransformersCharModel
from .voxprofile.src.model.emotion.wavlm_emotion import WavLMWrapper
from .voxprofile.src.model.emotion.whisper_emotion import WhisperWrapper
EMOTION_LABELS = [
    'Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 
    'Neutral', 'Sadness', 'Surprise', 'Other'
]

def load_SER_model(
    SER_model_name: str = "tiantiaf/wavlm-large-categorical-emotion",
    device: str = "auto",
    cache_dir: Optional[str] = None,
    model_batch_size: int = 16,
    backend: str = "auto",
    compute_type: Optional[str] = None,
) -> TransformersCharModel:
    """Initialise and return a emotion prediction model via Voxprofile."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "float32"

    if "wavlm" in SER_model_name:
        backend = "wavlm-large"
        model = WavLMWrapper.from_pretrained(SER_model_name).to(device)
    elif "whisper" in SER_model_name:
        backend = 'whisper'
        # model = WhisperWrapper.from_pretrained(SER_model_name).to(device)
        raise ValueError(f"Unmatch package version for model: {SER_model_name}. Need updates to fit in pipeline")
    else:
        raise ValueError(f"Unsupported model or backend: {SER_model_name}")
    model.eval() 
    if compute_type == "float16" and device == "cuda":
        model = model.half()
    return TransformersCharModel(
        backend=backend,
        model=model,
        Char_model_name=SER_model_name,
        device=device,
        cache_dir=cache_dir,
        model_batch_size=model_batch_size,
        compute_type=compute_type,
    )
    
    

def predict_emotion_segments(
    model: Any,
    segments: pd.DataFrame,
    output_dir: str,
    cache: bool = True,
    batch_size: Optional[float] = 30.0,
) -> Dict[str, Any]:
    """
    Slices segments into dynamic batches, verifies disk-cached files, 
    and passes uncached elements to WavLM batch inference before returning results.
    """
    batches = _batch_files(segments, output_dir, batch_size, max_duration_samples= 15.0)
    predictions_map = {}

    for batch in tqdm(batches, desc=f"Processing {len(batches)} emotions batches"):
        files_to_predict = []
        file_indices = []
        batch_results = [None] * len(batch)

        for i, seg in enumerate(batch):
            demo_cache = seg['seg_filename'].replace(".wav", "_emotions.txt")

            if cache and os.path.exists(demo_cache):
                try:
                    with open(demo_cache, "r", encoding="utf-8") as cache_file:
                        cached_text = cache_file.read().strip()
                    if not cached_text.startswith("[EMOTION_PREDICTION_FAILED:"):
                        parts = cached_text.split(" | ")
                        Catemo = float(parts[0].split(": ")[1])
                        batch_results[i] = {"EmoCat": Catemo}
                        continue
                except Exception:
                    pass  

            files_to_predict.append(seg)
            file_indices.append(i)

        # Model Inference execution block 
        if files_to_predict:
            outputs = _char_predict_batch_inference(files_to_predict, model, max_duration_samples=15.0)
            emotion_prob = F.softmax(outputs[0], dim=1)
            emo_indices = torch.argmax(emotion_prob, dim=1).cpu().tolist()
            # Map generated data targets back into batch metrics
            for batch_idx, pred_idx in zip(file_indices, emo_indices):
                current_emo = EMOTION_LABELS[pred_idx]

                batch_results[batch_idx] = {
                    "emoCat": current_emo
                }

                if cache:
                    # Construct cache file path smoothly
                    cache_path = batch[batch_idx].get("emo_cache", batch[batch_idx]['seg_filename'].replace(".wav", "_emotion.txt"))
                    with open(cache_path, "w", encoding="utf-8") as cache_file:
                        cache_file.write(f"emoCat: {current_emo}")

        # Save results back using their true global identifiers
        for seg, res in zip(batch, batch_results):
            predictions_map[seg["idx"]] = res

        # Clean up system memory after every batch iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return predictions_map