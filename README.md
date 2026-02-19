# RandomEmbedder Project

### Implementation Details

* A customizable `RandomEmbedder` class for prototyping and testing embedding workflows.
* Build your pipeline with modular, testable code:
  * `hash_to_vector()` for deterministic hashing-based vectors.
  * `generate_vector()` for random vector generation with optional normalization.
  * `embed()` for batch text-to-vector embedding.

### 1. **Text Input and Embedding**

* Accept a list of strings to be embedded.
* You can toggle between deterministic and random embeddings.
* Choose embedding dimensions (`dim`) and whether to normalize vectors.

### 2. **Vector Generation**

* For **deterministic mode**:
  * Use a hash-based approach to generate reproducible vectors via `hash_to_vector()`.

* For **random mode**:
  * Vectors are generated randomly via `generate_vector()`.
  * You can enable normalization for unit vectors.

### 3. **Embedding Interface**

* Use the `embed()` method to convert a list of strings into a NumPy array of vectors.
* All vectors will have shape `(len(texts), dim)`.

### ❗ Error Handling
* Ensure input texts are a list of strings. Type errors will be raised otherwise.
* Normalization is skipped if vector norm is zero (edge case handling in generate_vector())