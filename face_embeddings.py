import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
fa = FaceAnalysis(
    name="antelopev2/antelopev2",
    root = "/home/dsl/Desktop/chp/MultimodalEncoderVisualizer/",
    allowed_modules=['detection','recognition']
)
fa.prepare(ctx_id=0, det_size=(640, 640))
output_json = "antelopev2_embeddings.json"

def get_face_embedding(img_path):
    img =cv2.imread(img_path)
    faces= fa.get(img)

    if len(faces) < 1:
        print("No faces detected in the image. Skipping..")
        return None
    if len(faces) == 1:
        return faces[0].embedding
    if len(faces) >1:
        max_area = -1
        best_face = None
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            area = (x2-x1) * (y2 -y1)
            if area > max_area:
                max_area = area
                best_face = face
        
        return best_face.embedding

def main():

    records = []

    for person in os.listdir("clone_faces"):
        person_dir = os.path.join("clone_faces", person)
        for fn in os.listdir(person_dir):
            img_path = os.path.join(person_dir, fn)
            embedding = get_face_embedding(img_path)
            if embedding is None:
                continue
            records.append({
                "embedding": embedding.tolist(),
                "metadata": {
                    "name": person,
                    "image": img_path
                }
            })
    
    with open(output_json, "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"Wrote {len(records)} ebeddings to {output_json}")




def face_count() -> int:

    """ should equal the len(records)... which it does currently on Aug 7th 2025 without updates to clone_faces.
    7286 is the number of total faces in the directory"""
    count = 0
    for person in os.listdir("clone_faces"):
        person_dir = os.path.join("clone_faces", person)
        for fn in os.listdir(person_dir):
            count +=1

    return count









if __name__ == "__main__":
    main()
