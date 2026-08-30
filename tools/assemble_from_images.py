"""
使用已有的 images/ 和 audio.mp3/subtitle.srt 直接合成视频。
跳过脚本生成、配音、字幕、图片生成步骤，只做图片预处理 + 视频拼接。

用法:
    uv run python tools/assemble_from_images.py <task_id>

示例:
    uv run python tools/assemble_from_images.py 53da97ff-ed96-4bfe-8765-6e4ef8ae599e
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schema import MaterialInfo, VideoParams
from app.services import video as video_service
from app.utils import utils


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/assemble_from_images.py <task_id>")
        sys.exit(1)

    task_id = sys.argv[1]
    task_dir = utils.task_dir(task_id)
    print(f"Task directory: {task_dir}")

    # Load existing artifacts
    with open(os.path.join(task_dir, "script.json"), "r", encoding="utf-8") as f:
        script_data = json.load(f)
    with open(os.path.join(task_dir, "image_prompts.json"), "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)

    video_script = script_data["script"]
    params = VideoParams(**script_data["params"])

    # raw_prompts 已是 task.py 按 image_segment_mode 处理后的结果（merged 或 per_srt），
    # 若已带 _span 则不再二次合并，避免 6 -> 2 的误缩
    has_span = any(isinstance(p.get("_span"), (list, tuple)) for p in raw_prompts)
    if has_span:
        merged_prompts = raw_prompts
    else:
        merged_prompts = utils.merge_image_segments(raw_prompts, script=video_script)
    srt_path = os.path.join(task_dir, "subtitle.srt")
    srt_durations = utils.load_srt_durations(srt_path)

    print(f"Raw segments: {len(raw_prompts)}")
    print(f"Merged groups: {len(merged_prompts)}")
    print(f"SRT entries: {len(srt_durations)}")

    # Copy images to local_videos dir (preprocess_video requires paths inside it)
    local_videos_dir = utils.storage_dir("local_videos", create=True)
    image_dir = os.path.join(task_dir, "images")
    copied = 0

    materials = []
    for idx, item in enumerate(merged_prompts):
        span = item.get("_span")
        # 优先按 scene-00.png 顺序（WebUI 标准命名），兼容旧的序号命名
        scene_name = f"scene-{idx:02d}.png"
        src_path = os.path.join(image_dir, scene_name)
        if not os.path.exists(src_path):
            # 回退：按 _span 首索引猜旧命名
            img_idx = span[0] if span else idx
            img_num = img_idx + 1
            alt = os.path.join(image_dir, f"100 Years of Solitude_{img_num:05d}_.png")
            if os.path.exists(alt):
                src_path = alt
            else:
                # 最后回退：按目录排序的第 idx 张 png
                pngs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")]) if os.path.isdir(image_dir) else []
                if idx < len(pngs):
                    src_path = os.path.join(image_dir, pngs[idx])
                else:
                    print(f"WARNING: Image not found: {src_path}")
                    continue

        if not os.path.exists(src_path):
            print(f"WARNING: Image not found: {src_path}")
            continue

        # Copy to local_videos dir with predictable name
        dest_name = f"scene-{copied:02d}.png"
        dest_path = os.path.join(local_videos_dir, dest_name)
        shutil.copy2(src_path, dest_path)
        copied += 1

        group_duration = 0.0
        if span and srt_durations:
            start_idx, end_idx = span
            if 0 <= start_idx < len(srt_durations) and 0 <= end_idx < len(srt_durations):
                group_duration = sum(
                    ed - st
                    for st, ed in srt_durations[start_idx : end_idx + 1]
                )

        materials.append(
            MaterialInfo(
                provider="ai_image",
                url=dest_path,
                duration=int(round(group_duration)) or params.video_clip_duration,
                source_info={
                    "provider": "ai_image",
                    "prompt": item.get("prompt", ""),
                    "segment": item.get("segment", ""),
                    "_span": span,
                },
            )
        )

    print(f"Materials: {len(materials)}")
    for i, m in enumerate(materials):
        span = m.source_info.get("_span", [])
        print(f"  [{i:2d}] span={str(span):12s} dur={m.duration}s  {os.path.basename(m.url)}")

    # Preprocess images to video clips (adds zoom effect, uses per-material duration)
    processed = video_service.preprocess_video(
        materials, clip_duration=params.video_clip_duration
    )
    video_paths = [m.url for m in processed if m.url]
    print(f"Processed clips: {len(video_paths)}")

    # Load audio
    audio_file = os.path.join(task_dir, "audio.mp3")
    from moviepy import AudioFileClip

    audio_clip = AudioFileClip(audio_file)
    audio_duration = audio_clip.duration
    audio_clip.close()
    print(f"Audio duration: {audio_duration:.2f}s")

    # Combine videos
    combined_path = os.path.join(task_dir, "combined-1.mp4")
    print(f"Combining video => {combined_path}")
    video_service.combine_videos(
        combined_video_path=combined_path,
        video_paths=video_paths,
        audio_file=audio_file,
        video_aspect=params.video_aspect,
        video_concat_mode=params.video_concat_mode,
        video_transition_mode=params.video_transition_mode,
        max_clip_duration=params.video_clip_duration,
        threads=params.n_threads,
        clip_speed=params.video_clip_speed,
    )
    print(f"Combined video: {combined_path}")

    if not os.path.exists(combined_path) or os.path.getsize(combined_path) == 0:
        print("ERROR: combined video not created or empty")
        sys.exit(1)

    # Generate final video with subtitles
    final_path = os.path.join(task_dir, "final-1.mp4")
    print(f"Generating final video => {final_path}")
    bgm_mix_succeeded = video_service.generate_video(
        video_path=combined_path,
        audio_path=audio_file,
        subtitle_path=srt_path,
        output_file=final_path,
        params=params,
        bgm_file_override="",
    )
    print(f"Final video: {final_path}")
    if os.path.exists(final_path):
        size_mb = os.path.getsize(final_path) / 1024 / 1024
        print(f"Done! Video size: {size_mb:.1f} MB")
    else:
        print("ERROR: Final video not generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
