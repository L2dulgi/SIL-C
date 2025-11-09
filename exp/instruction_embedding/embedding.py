import json
import pickle
import openai
import numpy as np
import os
from SILGym.utils.logger import get_logger

logger = get_logger(__name__)

# Retrieve your OpenAI API key from the environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def load_instructions(json_file):
    """
    Load instructions from a JSON file.
    The JSON file should contain a dictionary mapping skills to instructions.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_embedding(text, model="text-embedding-3-large", dimensions=None):
    """
    Get the embedding vector for the given text using OpenAI's embedding model.
    Allows specifying the embedding dimensions.
    """
    response = openai.embeddings.create(
        input=[text],
        model=model,
        dimensions=dimensions
    )
    embedding = response.data[0].embedding
    return embedding

def save_embeddings(embeddings, pkl_file):
    """
    Save the embeddings dictionary to a pickle file.
    """
    with open(pkl_file, "wb") as f:
        pickle.dump(embeddings, f)

def check_embedding_dimensions(embeddings):
    """
    Print the dimension of each embedding vector in the embeddings dictionary.
    """
    for key, vector in embeddings.items():
        vector_np = np.array(vector)
        logger.info(f"{key}: {vector_np.shape}")

def embedding_extract():
    # Load instructions from JSON file
    json_file = "exp/instruction_embedding/kitchen.json"
    json_file = "exp/instruction_embedding/mmworld.json"
    instructions = load_instructions(json_file)
    
    # Define the list of dimensions
    dimensions_list = [256, 512, 768, 1536, 3072]
    
    for dimensions in dimensions_list:
        # Generate embeddings for each instruction with the specified dimensions
        embeddings_dict = {}
        for skill, instruction in instructions.items():
            embedding_vector = get_embedding(instruction, dimensions=dimensions)
            embeddings_dict[skill] = embedding_vector
            logger.info(f"Generated {dimensions}-dimensional embedding for '{skill}'.")
        
        # Save the embeddings dictionary to a pickle file
        pkl_file = f"exp/instruction_embedding/mmworld/{dimensions}.pkl"
        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        save_embeddings(embeddings_dict, pkl_file)
        logger.info(f"Embeddings saved to '{pkl_file}'.")
        
        # Check and print the dimensions of the embeddings
        logger.info(f"Checking dimensions of embeddings in '{pkl_file}':")
        check_embedding_dimensions(embeddings_dict)
        logger.info("\n" + "="*50 + "\n")


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_embeddings(pkl_file):
    """
    Load embeddings from a pickle file.
    """
    with open(pkl_file, "rb") as f:
        return pickle.load(f)

def compute_cosine_similarity(embeddings_matrix):
    """
    Compute the cosine similarity matrix for the given embeddings.
    """
    norm = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized = embeddings_matrix / norm
    similarity = np.dot(normalized, normalized.T)
    return similarity

def plot_similarity_map(similarity, labels, save_path):
    """
    Plot and save a heatmap of the cosine similarity matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Embedding Cosine Similarity")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved similarity map to {save_path}")

def main():
    # Target directory and dimensionalities
    dim_list = [256, 512, 768, 1536, 3072]
    input_dir = "exp/instruction_embedding/kitchen"
    input_dir = "exp/instruction_embedding/mmworld"
    output_dir = "data/embeddings"

    for dim in dim_list:
        pkl_path = os.path.join(input_dir, f"{dim}.pkl")
        if not os.path.exists(pkl_path):
            logger.warning(f"File not found: {pkl_path}")
            continue

        embeddings = load_embeddings(pkl_path)
        labels = list(embeddings.keys())
        embeddings_matrix = np.array([embeddings[label] for label in labels])
        similarity = compute_cosine_similarity(embeddings_matrix)
        save_path = os.path.join(output_dir, f"{dim}_similarity.png")
        plot_similarity_map(similarity, labels, save_path)

if __name__ == "__main__":
    embedding_extract()
    main()

