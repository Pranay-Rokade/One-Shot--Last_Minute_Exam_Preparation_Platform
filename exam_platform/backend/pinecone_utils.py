import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)


def get_all_context_from_pinecone() -> str:
    """Retrieve all stored context text from Pinecone index (v4+ safe)."""
    index = pc.Index(INDEX_NAME)
    print("Retrieving all context from Pinecone index...")

    # list() yields pages of IDs
    id_pages = index.list()
    all_ids = []
    for page in id_pages:
        all_ids.extend(page)  # flatten each page

    print(f"Found {len(all_ids)} total vectors. Fetching metadata in batches...")

    context_list = []
    batch_size = 20  # safe batch size to avoid long URLs

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        try:
            fetched = index.fetch(ids=batch_ids)
            for _id, record in fetched.vectors.items():
                metadata = getattr(record, "metadata", None)
                if metadata and "context" in metadata:   # 👈 FIXED HERE
                    context_list.append(metadata["context"])
        except Exception as e:
            print(f"⚠️ Error fetching batch {i//batch_size+1}: {e}")

    combined_context = "\n".join(context_list)
    print(f"✅ Retrieved {len(context_list)} text chunks.")
    return combined_context.strip()


if __name__ == "__main__":
    context = get_all_context_from_pinecone()
    print(context[:1000])
