import inspect
import logging
import streamlit as st
from src.embedding import randomEmbedder

st.set_page_config(page_title="🔧 Random Embedder", layout="centered")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("random_embedder")

def log_and_show_error(e: Exception, placeholder):
    logger.error(e, exc_info=True)
    placeholder.error(f"❌ Error: {str(e)}")


def main():
    st.title("🔧 Random Embedding Generator")
    st.write(
        "Welcome to your **developer's playground** — a quick way to generate dummy embeddings for testing your ML pipelines."
    )

    embedding_dim = st.slider("Select Embedding Dimension (vector size):", 16, 512, 128, step=16)
    st.caption("ℹ️ Embedding dimension must be greater than 0. Minimum is 16.")
    deterministic = st.checkbox("Enable Deterministic Mode (repeatable embeddings)", value=True)
    text_input = st.text_area("Enter text chunks (one per line):", height=200)
    try:
        if st.button("🚀 Generate Embeddings"):
                status_placeholder = st.empty()
                embedder = randomEmbedder(dim=embedding_dim, deterministic=deterministic)
                lines = [line.strip() for line in text_input.split("\n") if line.strip()]
                if not lines:
                    st.warning("⚠️ Please enter at least one line of text to embed.")
                    return

                status_placeholder.info("🛠️ Generating embeddings...")
                embeddings = embedder.embed(lines)
                status_placeholder.success("✅ Embeddings generated successfully!")

                st.write("### Embedding Matrix Shape")
                st.code(f"{embeddings.shape}  # (number_of_texts, embedding_dimension)")

                st.write("### Sample Embeddings (first 3 rows)")
                st.dataframe(embeddings[:3])

    except NotImplementedError as e :
        status_placeholder.warning(f'⚠️ Please {e} before testing.')

    except Exception as e:
        log_and_show_error(e,status_placeholder)


if __name__ == "__main__":
    main()
