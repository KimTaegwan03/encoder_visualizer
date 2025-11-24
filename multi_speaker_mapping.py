import os, json
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from numpy import array
from utils.utils import *
import base64
from io import BytesIO
from PIL import Image
from collections import defaultdict
from trash.active_speaker_match import generate_query_embedding_from_ndarray, get_max_similarity_one
from faceDB import search, insert_records, load_json, ensure_collection, connect
import torch

# --- 경로 설정 ---
orch_dir = "/mnt/data1/orch_jsons"
asd_dir = "/mnt/data1/asd_bbox_per_sec_jsons"
video_dir = "/mnt/data1/videos_25fps"
dest_dir = "/mnt/data1/asd_orch_jsons"

def relabel_orch_speakers(orch_file: str, mapping: dict[str, str], dest_file: str | None):
    """Orchestration JSON 파일의 화자 레이블을 변경하여 새 파일로 저장합니다."""
    with open(orch_file, "r", encoding ="utf-8") as f:
        data = json.load(f)

    for seg in data:
        spk = seg.get("speaker")
        if spk in mapping:
            seg["speaker"] = mapping[spk]
        
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_video(video_path: str) -> str:
    """
    단일 비디오 파일을 처리하는 함수 (멀티프로세싱의 작업 단위).
    
    Args:
        video_path (str): 처리할 비디오 파일의 이름.

    Returns:
        str: 처리 결과를 나타내는 상태 메시지.
    """
    client = connect()  # 각 프로세스마다 새로운 클라이언트 연결 생성
    try:
        coll = ensure_collection(client)
        basename = os.path.basename(video_path).replace(".mp4","")
        proper_video_path = os.path.join(video_dir, video_path)
        orch_file_path = os.path.join(orch_dir, basename + ".json")
        asd_file_path = os.path.join(asd_dir, basename + ".json")
        dest_file_path = os.path.join(dest_dir, basename + ".json")

        if os.path.isfile(dest_file_path):
            return f"Skipped: {basename} (already exists)"

        overlap = merge_asr_asd(orch_file_path, asd_file_path)
        if overlap is None:
            return f"Skipped: {basename} (no overlap or empty orch file)"

        imgs = extract_faces_from_video(proper_video_path, overlap)
        
        # 화자(speaker)별로 인식된 얼굴 이름(name)을 저장
        speaker_to_names = defaultdict(list)
        for img in imgs:
            query_embedding = generate_query_embedding_from_ndarray(img["image"])
            _, filtered = search(coll, query_embedding, 0.5)
            best = get_max_similarity_one(filtered)
            
            recognized_name = "unknown"
            if best:
                recognized_name = best["name"]
            
            speaker_to_names[img["speaker"]].append(recognized_name)

        # 각 화자에 대해 가장 많이 인식된 이름으로 최종 매핑 생성
        speaker_name_mapping = {}
        for spk, names in speaker_to_names.items():
            if not names:
                continue
            counts = Counter(names)
            # 'unknown'이 가장 많더라도 다른 이름이 있다면 그 이름을 우선 사용
            if counts.most_common(1)[0][0] == 'unknown' and len(counts) > 1:
                desired_name = counts.most_common(2)[1][0]
            else:
                desired_name = counts.most_common(1)[0][0]
            speaker_name_mapping[spk] = desired_name
        
        # 최종 매핑을 사용하여 JSON 파일 다시 쓰기
        relabel_orch_speakers(orch_file_path, speaker_name_mapping, dest_file_path)
        
        return f"Success: {basename}"
    except Exception as e:
        # 에러 발생 시 로그를 남기고 다음 파일 처리 계속
        return f"Error processing {video_path}: {e}"
    finally:
        client.close()  # 작업 완료 후 반드시 클라이언트 연결 종료

def main():
    """메인 실행 함수"""
    # 1) Weaviate에 얼굴 임베딩 데이터 로드 (최초 한 번만 실행)
    with connect() as client:
        coll = ensure_collection(client)
        records = load_json("antelopev2_embeddings.json")
        insert_records(coll, records)

    # 2) 비디오 목록 가져오기 및 정렬
    videos = os.listdir(video_dir)
    videos.sort()

    # 3) 리스트의 중앙에서부터 바깥쪽으로 처리하도록 순서 재정렬
    n = len(videos)
    reordered_list = []
    left, right = (n - 1) // 2, (n - 1) // 2 + 1
    while left >= 0 or right < n:
        if left >= 0:
            reordered_list.append(videos[left])
            left -= 1
        if right < n:
            reordered_list.append(videos[right])
            right += 1

    # 4) 멀티프로세싱으로 비디오 처리
    # GPU 메모리 상황에 맞춰 max_workers 조절 (예: 2-4)
    # CPU만 사용 시 os.cpu_count() // 2 정도로 설정
    max_workers = 3
    
    print(f"Starting video processing with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map을 tqdm으로 감싸 전체 진행률 표시
        results = list(tqdm(executor.map(process_video, reordered_list), total=len(reordered_list)))

    # 5) 결과 요약
    success_count = sum(1 for r in results if r.startswith("Success"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    error_count = sum(1 for r in results if r.startswith("Error"))

    print("\n--- Processing Summary ---")
    print(f"Total videos: {len(videos)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    
    if error_count > 0:
        print("\n--- Error Details ---")
        for r in results:
            if r.startswith("Error"):
                print(r)

if __name__ == "__main__":
    # PyTorch GPU 모델을 사용하는 멀티프로세싱의 경우 'spawn' 시작 방식 권장
    # CUDA 초기화 관련 오류 방지
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
