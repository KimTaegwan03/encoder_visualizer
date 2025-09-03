import os, json
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from numpy import array
from utils.utils import *
import base64
from io import BytesIO
from PIL import Image
from collections import defaultdict
from active_speaker_match import generate_query_embedding_from_ndarray, get_max_similarity_one
from faceDB import search, insert_records, load_json, ensure_collection, connect

orch_dir = "/mnt/data1/orch_jsons"
asd_dir = "/mnt/data1/asd_jsons"
video_dir = "/mnt/data1/videos_25fps"
dest_dir = "/mnt/data1/asd_orch_jsons"
def relabel_orch_speakers(orch_file: str, mapping: dict[str, str], dest_file: str | None):
    with open(orch_file, "r", encoding ="utf-8") as f:
        data = json.load(f)

    for seg in data:
        spk = seg.get("speaker")
        if spk in mapping:
            seg["speaker"] = mapping[spk]
        
        # else:
        #     seg["speaker"] = "unknown"
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def main():
    coll = ensure_collection(client)
    # 1) Load embeddings JSON
    records = load_json("antelopev2_embeddings.json")
    insert_records(coll, records)
    #read video
    count =0
    videos = os.listdir(video_dir)
    videos.sort()
    for video_path in tqdm(videos):
        try:
            basename = os.path.basename(video_path).replace(".mp4","")
            proper_video_path = os.path.join(video_dir, video_path)
            orch_file_path = os.path.join(orch_dir,basename+".json")
            asd_file_path = os.path.join(asd_dir,basename+".json")
            dest_file_path = os.path.join(dest_dir,basename+".json")
            if os.path.isfile(dest_file_path):
                print(f"File {dest_file_path} already exists. Skipping...")
                continue
            overlap = merge_asr_asd(orch_file_path,asd_file_path)
            if overlap is None:
                continue #empty orch files 
            imgs = extract_faces_from_video(proper_video_path, overlap)    
            dic = {}
            for img in imgs:
                query_embedding = generate_query_embedding_from_ndarray(img["image"])
                _, filtered = search(coll, query_embedding, 0.5)
                best = get_max_similarity_one(filtered)
                if best:
                    # img["speaker"] = best["name"]
                    img["name"] = (best["name"])
                    img["retrieved"] = best["image_path"]
                    img["score"] = best["cosine_similarity"]
                    if img["speaker"] not in dic.keys():
                        dic[img["speaker"]] = [] 
                    dic[img["speaker"]].append(img["name"]) 
                else:
                    # img["speaker"] = "unknown"
                    img["name"] = "unknown"
                    img["retrieved"] = None
                    img["score"] = 0.0    
                    if img["speaker"] not in dic.keys():
                        dic[img["speaker"]] = [] 
                    dic[img["speaker"]].append(img["speaker"])
            dic2 = {} 
            for spk in dic.keys():
                counts = Counter(dic[spk])
                desired_name = max(counts, key=counts.get)
                dic2[spk] = desired_name
            #save

            relabel_orch_speakers(orch_file_path, dic2, dest_file_path)
            for img in imgs:
                img["speaker"] = dic2[img["speaker"]]
            count+=1
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue
    print(f"All {count} videos processed successfully.")  
                


if __name__ =="__main__":
    client = connect()
    try:
        main()
    finally:
        client.close()