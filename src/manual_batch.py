from typing import List, Dict, Any, Optional
import os
import pandas as pd
def _batch_files(
    segments: pd.DataFrame,
    output_dir: str,
    batch_size: Optional[float] = 30.0,
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
        float(batch_size) if batch_size and batch_size > 0 else float("inf")
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