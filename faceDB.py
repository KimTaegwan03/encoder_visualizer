import os 
import cv2
from insightface.app import FaceAnalysis
import os, json
import weaviate
from uuid import uuid4
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.util import generate_uuid5
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
fa = FaceAnalysis(
    name="antelopev2/antelopev2",
    root = "/home/dsl/Desktop/chp/MultimodalEncoderVisualizer/",
    allowed_modules=['detection','recognition']
)
fa.prepare(ctx_id=0, det_size=(640, 640))
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "http://localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
COLL_NAME = "FaceEmbeddings"

def connect():
    # Local or remote; add api keys if using WCS
    return weaviate.connect_to_local()

def ensure_collection(client):
    # Drop & recreate if you want a fresh start:
    # client.collections.delete(COLL_NAME)

    if COLL_NAME not in client.collections.list_all():
        client.collections.create(
            name=COLL_NAME,
            vector_config=Configure.Vectors.self_provided(
                name="default",
                # HNSW + cosine is good for face embeddings
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(name="image_path", data_type=DataType.TEXT),
            ],
        )
    return client.collections.get(COLL_NAME)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def insert_records(coll, records, use_deterministic_ids=True):
    """
    records: list of { "embedding": [floats...], "metadata": {"name": str, "image": str} }
    """
    objs = []
    for r in records:
        props = {
            "name": r["metadata"]["name"],
            "image_path": r["metadata"]["image"],
        }
        # Deterministic UUID helps avoid duplicates if you re-run
        obj_id = generate_uuid5(props) if use_deterministic_ids else str(uuid4())
        obj = DataObject(
            properties=props,
            vector=r["embedding"],
            uuid=obj_id
        )
        objs.append(obj)
    # Insert in batches automatically (client handles chunking internally)
    coll.data.insert_many(objs)
def generate_query_embedding(cropped_image):
    img = cv2.imread(cropped_image)
    if img is None:
        print(f"Failed to load image at {cropped_image}")
        return None
    
    img = cv2.resize(img, (112,112))
    img = img.astype("float32")

    return fa.models['recognition'].get_feat(img).flatten().tolist()
def search(
    coll,
    query_embedding,
    min_cosine_sim=0.65,   # e.g., keep only >= 0.35 similarity
):
    """
    Perform a cosine-similarity based vector search using weaviate v4
    Returns (results, filtered) - where filtered applies similarity threshold
    """
    res = coll.query.near_vector(
        near_vector=query_embedding,
        return_metadata=MetadataQuery(distance=True)  # we’ll convert to sim
    )
    results = []
    for obj in res.objects:
        dist = obj.metadata.distance
        cosine_similarity = 1.0 - dist if dist is not None else None
        results.append({
            "uuid": str(obj.uuid),
            "name": obj.properties.get("name"),
            "image_path": obj.properties.get("image_path"),
            "distance": dist,
            "cosine_similarity": cosine_similarity,
        })
    filtered = [r for r in results if r["cosine_similarity"] is not None and r["cosine_similarity"] >= min_cosine_sim]
    return results, filtered

if __name__ == "__main__":
    client = connect()
    try:

        coll = ensure_collection(client)

        # 1) Load your embeddings JSON (format described above)
        records = load_json("antelopev2_embeddings.json")
        insert_records(coll, records)

        query_embedding = generate_query_embedding("download.jpeg")
        results, filtered = search(coll, query_embedding, 0.5)
        if filtered:
            top = filtered[0]
            print(f"Best match: {top['name']} ({top['cosine_similarity']:.4f})")
            print(f"Image path: {top['image_path']}")
        if not filtered:
            print("No matches found above threshold.")

    finally:
        client.close()