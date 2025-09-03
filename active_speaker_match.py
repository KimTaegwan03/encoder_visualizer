from insightface.app import FaceAnalysis
import os
from faceDB import search, insert_records, load_json, ensure_collection, connect
from insightface_app import compare_faces, get_face_embedding
from utils.utils import extract_faces_from_video, merge_asr_asd
import tempfile
import cv2
json_dir = "/mnt/data1/orch_jsons"
video_dir = "/mnt/data1/videos"
# audio_dir = "data/audio"
feature_dir = "/mnt/data1/emb"
graph_dir = "/mnt/data1/emb/graph"
client = connect()
fa = FaceAnalysis(
    name="antelopev2/antelopev2",
    root = "/home/dsl/Desktop/chp/MultimodalEncoderVisualizer/",
    allowed_modules=['detection','recognition']
)
fa.prepare(ctx_id=0, det_size=(640, 640))

def generate_query_embedding(cropped_image):
    img = cv2.imread(cropped_image)
    if img is None:
        print(f"Failed to load image at {cropped_image}")
        return None
    
    img = cv2.resize(img, (112,112))
    img = img.astype("float32")

    return fa.models['recognition'].get_feat(img).flatten().tolist()

def generate_query_embedding_from_ndarray(img):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, img)

    embedding = generate_query_embedding(temp_path)

    os.remove(temp_path)
    return embedding

def get_max_similarity_one(filtered):
    if filtered:
        max = filtered[0]['cosine_similarity']
        best =filtered[0]
        for entry in filtered[1:]:
            if max < entry['cosine_similarity']:
                max = entry['cosine_similarity']
                best = entry
        return best
    else: 
        return None


def main():
    try:
        coll = ensure_collection(client)

        # 1) Load embeddings JSON (format described above)
        records = load_json("antelopev2_embeddings.json")
        insert_records(coll, records)
        query_embedding = generate_query_embedding("download.jpeg")
        _, filtered = search(coll, query_embedding, 0.5)
        if filtered:
            top = filtered[0]
            print(f"Individual: {top['name']} \nImage: {top['image_path']} \nScore: {top['cosine_similarity']:.4f}")
        else:
            print("No matching individual found.")
    finally:
        client.close()

if __name__ =="__main__":
    main()