from src.embedding import randomEmbedder

texts = ["hello world", "test text", "another one"]
embedder = randomEmbedder(dim=64, deterministic=True)

vectors = embedder.embed(texts)
print(vectors.shape)  # (3, 64)