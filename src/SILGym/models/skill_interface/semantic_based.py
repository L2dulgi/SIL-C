import numpy as np
import random
from SILGym.utils.cuml_wrapper import KMeans
import matplotlib.pyplot as plt
from SILGym.models.skill_interface.base import BaseInterface
from SILGym.utils.logger import get_logger

class ImanipInterfaceConfig :
    def __init__(self, verbose=False, visualize=False):
        self.verbose = verbose
        self.visualize = visualize
        self.unknown_entry_option = "previous" # ["random", "previous", "default"]

class IsCiLInterfaceConfig:
    """
    Configuration class for Interface.
    
    Parameters:
      - bases_num: number of clusters (centroids) to create per skill.
      - verbose: if True, print detailed log messages.
      - visualize: if True, visualize each new prototype matrix when it is created.
    """
    def __init__(self, bases_num=5, verbose=False, visualize=False):
        self.bases_num = bases_num
        self.verbose = verbose
        self.visualize = visualize

class SemanticInterface(BaseInterface):
    def __init__(self, config, semantic_emb_path="exp/instruction_embedding/kitchen/512.pkl"):
        """
        Initialize the SemanticInterface.
        
        Parameters:
          - config: a configuration object (can contain verbose flags, unknown_entry_option, default_entry, etc.)
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.config = config

        # Load semantic embeddings from the 512-d embedding file.
        try:
            with open(semantic_emb_path, "rb") as f:
                self.semantic_embeddings = pickle.load(f)
                for key, value in self.semantic_embeddings.items():
                    if isinstance(value, list):
                        self.semantic_embeddings[key] = np.array(value)
                self.semantic_dim = len(list(self.semantic_embeddings.values())[0])
                self.logger.info(f"Loaded {len(self.semantic_embeddings)} semantic embeddings with dimension {self.semantic_dim}")
        except Exception as e:
            raise RuntimeError(f"Failed to load semantic embeddings from {semantic_emb_path}: {e}")

        # Dictionaries to map skill strings to unique integer IDs and vice versa.
        self.skill_to_id = {}
        self.entry_to_skill = {}

        # NOTE : preregistering is not feasible for real-world applications.
        self.pre_register = False 
        if self.pre_register == True:
            for idx, skill in enumerate(self.semantic_embeddings.keys()):
                self.skill_to_id[skill]   = idx
                self.entry_to_skill[idx]  = skill
            self.logger.info(f"preregistered {len(self.skill_to_id)} skills")
            # Stores the last valid entry encountered (for 'previous' option).
        self.last_valid_entry = None

    @property
    def num_skills(self):
        """
        Return the number of unique skills currently managed by the interface.
        """
        return len(self.skill_to_id.keys())

    def init_entry(self, dataloader):
        """
        Initialization step: Create or reset necessary fields in dataloader.stacked_data.
        
        This method initializes the 'entry' and 'skill_id' fields.
        """
        if not hasattr(dataloader, "stacked_data"):
            raise ValueError("Dataloader must have a 'stacked_data' attribute")
        data = dataloader.stacked_data
        # Assume observations is a numpy array; use its length as the number of samples.
        T = len(data.get("observations", []))
        data["entry"] = np.zeros((T,), dtype=np.int32)
        data["skill_id"] = np.zeros((T,), dtype=np.int32)

        data["skill_aux"] = np.zeros((T, self.semantic_dim), dtype=np.float32)  
        for i in range(T):
            data['skill_aux'][i] = self.semantic_embeddings.get(
                data['skills'][i], np.zeros(self.semantic_dim)
            )
        self.logger.info(f"skill_aux initialized with shape {data['skill_aux'].shape}")
        
        dataloader.stacked_data = data
        return dataloader

    def map_entry(self, dataloader):
        """
        Main mapping logic: Map each skill to a fixed entry.
        
        This method uses the 'skills' field in dataloader.stacked_data to build a mapping
        from skill strings to unique integer IDs, and then populates the 'entry' and 'skill_id'
        fields accordingly.
        """
        data = dataloader.stacked_data
        if "skills" not in data:
            raise ValueError("Stacked data must contain a 'skills' field with string labels")
        skills = data["skills"]  # Assume a numpy array of strings
        unique_skills = np.unique(skills)
        
        # Build mapping for each unique skill.
        for skill in unique_skills:
            if skill not in self.skill_to_id:
                new_id = len(self.skill_to_id)
                self.skill_to_id[skill] = new_id
                self.entry_to_skill[new_id] = skill
                if self.config.verbose:
                    self.logger.info(f"New skill detected: '{skill}' assigned id {new_id}")
        
        # Update the 'entry' and 'skill_id' fields based on the mapping.
        T = len(skills)
        entries = np.zeros((T,), dtype=np.int32)
        skill_ids = np.zeros((T,), dtype=np.int32)
        for i, skill in enumerate(skills):
            skill_id = self.skill_to_id[skill]
            entries[i] = skill_id
            skill_ids[i] = skill_id
        data["entry"] = entries
        data["skill_id"] = skill_ids
        data["decoder_id"] = skill_ids.copy()  # For imanip
        return dataloader

    def update_interface(self, dataloader):
        """
        Entry point that orchestrates the interface processing.
        
        This method resets the interface and then initializes and maps the dataloader entries.
        """
        dataloader = self.init_entry(dataloader)
        dataloader = self.map_entry(dataloader)
        return self.update_dataloader(dataloader)

    def update_dataloader(self, dataloader):
        """
        Update the dataloader with new data.
        
        For the SemanticInterface, no additional modifications are needed.
        """
        dataloader.stacked_data['orig_obs'] = dataloader.stacked_data['observations'].copy()
        dataloader.stacked_data['observations'] = np.concatenate(
            (dataloader.stacked_data['observations'], dataloader.stacked_data['skill_aux']), axis=-1
        )
        return dataloader

    def rollback_dataloader(self, dataloader):
        """
        Rollback the dataloader to the previous state.
        
        For the SemanticInterface, no changes are made, so the dataloader is returned unchanged.
        """
        if 'orig_obs' not in dataloader.stacked_data:
            return dataloader
        dataloader.stacked_data['observations'] = dataloader.stacked_data['orig_obs']
        del dataloader.stacked_data['orig_obs']
        return dataloader

    def forward(self, entry, current_state=None):
        """
        Forward pass of the interface.
        
        In this simple semantic mapping, the forward pass returns the input entry as both
        the skill id and the auxiliary output. If an unseen entry is encountered, it is replaced
        according to the configuration option: 'random', 'previous', or 'default'.
        """
        if np.isscalar(entry):
            entry = np.array([entry], dtype=np.int32)
        else:
            entry = np.array(entry, dtype=np.int32)
            
        output = []
        for e in entry:
            # Check if the entry is known based on the entry_to_skill mapping.
            if e in self.entry_to_skill:
                valid_entry = e
                self.last_valid_entry = e
            else:
                # Get option for unknown entry handling; default to 'default' if not provided.
                option = getattr(self.config, "unknown_entry_option", "default")
                if option == "random":
                    valid_keys = list(self.entry_to_skill.keys())
                    if valid_keys:
                        valid_entry = random.choice(valid_keys)
                    else:
                        valid_entry = getattr(self.config, "default_entry", 0)
                elif option == "previous":
                    if self.last_valid_entry is not None:
                        valid_entry = self.last_valid_entry
                    else:
                        valid_entry = getattr(self.config, "default_entry", 0)
                else:  # 'default'
                    valid_entry = getattr(self.config, "default_entry", 0)
            output.append(valid_entry)
        output = np.array(output, dtype=np.int32)
        entry = output.copy()
        decoder_ids = output.copy()  

        # For the SemanticInterface, the auxiliary output is the semantic embedding corresponding to the skill.
        semantic_output = []
        for e in entry:
            skill_name = self.entry_to_skill.get(e, "Unknown")
            if skill_name not in self.semantic_embeddings:
                raise ValueError(f"Semantic embedding for skill '{skill_name}' not found")
            semantic_output.append(self.semantic_embeddings[skill_name])
        semantic_output = np.array(semantic_output)

        return (entry, decoder_ids), semantic_output 

    def get_skill_ids(self, skills) :
        """
        Given a list of skills, return their corresponding skill ids.
        
        Parameters:
            skills: A list of skill strings.
        
        Returns:
            A list of skill ids corresponding to the input skills.
        """
        skill_ids = []
        for skill in skills:
            if skill in self.skill_to_id:
                skill_ids.append(self.skill_to_id[skill])
            else:
                skill_ids.append(0) # unknown skill to 0(default)
        return np.array(skill_ids)

    # -----------------------------------------
    # Replay method for Imanip implementation
    # -----------------------------------------
    def sample_rehearsal_old(self, new_data_dict, keep_ratio):
        """
        Sample a fraction ('keep_ratio') of the 'observations' from new_data_dict.
        """
        if 'observations' in new_data_dict and len(new_data_dict['observations']) > 0:
            new_size = len(new_data_dict['observations'])
            keep_count = int(new_size * keep_ratio)
            self.logger.info(f"New data size = {new_size}, keeping {keep_count} items.")

            if keep_count > 0:
                indices = np.random.choice(new_size, keep_count, replace=False)
                return {k: v[indices] for k, v in new_data_dict.items()}
            else:
                self.logger.warning("keep_ratio=0, no new data kept.")
                return None
        else:
            self.logger.warning("No 'observations' or empty new data.")
            return None
        
    def sample_rehearsal(self, new_data_dict, keep_ratio):
        """
        Sample a fraction ('keep_ratio') of the 'observations' from new_data_dict.
        """
        if 'observations' in new_data_dict and len(new_data_dict['observations']) > 0:
            new_size = len(new_data_dict['observations'])
            keep_count = int(new_size * keep_ratio)
            self.logger.info(f"New data size = {new_size}, keeping {keep_count} items.")

            if keep_count > 0:
                indices = np.random.choice(new_size, keep_count, replace=False)
                sampled_dataset = {k: v[indices] for k, v in new_data_dict.items()}
            else:
                self.logger.warning("keep_ratio=0, no new data kept.")
                sampled_dataset = None
        else:
            self.logger.warning("No 'observations' or empty new data.")
            sampled_dataset = None
            
        return sampled_dataset
import pickle

class PrototypeInterface(BaseInterface):
    def __init__(self, config: IsCiLInterfaceConfig, semantic_emb_path="exp/instruction_embedding/kitchen/512.pkl"):
        """
        Initialize the PrototypeInterface and load the 512-d semantic embeddings.
        
        Parameters:
            config: Instance of IsCiLInterfaceConfig containing necessary settings.
            semantic_emb_path: Path to the pickle file containing 512-d embeddings.
        """
        self.logger = get_logger(__name__)
        self.config = config
        self.bases_num = config.bases_num if hasattr(config, 'bases_num') else 5

        # Load semantic embeddings from the 512-d embedding file.
        try:
            with open(semantic_emb_path, "rb") as f:
                self.semantic_embeddings = pickle.load(f)
                for key, value in self.semantic_embeddings.items():
                    if isinstance(value, list):
                        self.semantic_embeddings[key] = np.array(value)
                self.semantic_dim = len(list(self.semantic_embeddings.values())[0])
                self.logger.info(f"Loaded {len(self.semantic_embeddings)} semantic embeddings with dimension {self.semantic_dim}")
        except Exception as e:
            raise RuntimeError(f"Failed to load semantic embeddings from {semantic_emb_path}: {e}")

        # Initialize mapping dictionaries and prototype storage.
        self.skill_to_id = {}      # Key: skill string, value: unique integer id (entry)
        self.entry_to_skill = {}     # Key: unique integer id, value: skill string
        self.skill_prototypes = {}   # Key: skill id, value: list of prototypes (tuple of (prototype_id, centroid_matrix))

        # NOTE : preregistering is not feasible for real-world applications.
        self.pre_register = False 
        if self.pre_register == True:
            for idx, skill in enumerate(self.semantic_embeddings.keys()):
                self.skill_to_id[skill]    = idx
                self.entry_to_skill[idx]   = skill
                self.skill_prototypes[idx] = []     
            self.logger.info(f"preregistered {len(self.skill_to_id)} skills")
        self.next_prototype_id = 0   # Global counter for new prototype IDs

    @property
    def num_skills(self):
        """
        Return the number of unique skills currently managed by the interface.
        """
        return len(self.skill_to_id.keys())

    def init_entry(self, dataloader):
        """
        Initialize the 'entry', 'skill_id', and 'skill_aux' fields in dataloader.stacked_data
        if they do not already exist.
        """
        stacked_data = dataloader.stacked_data
        T = len(stacked_data['observations'])
        if 'entry' not in stacked_data:
            stacked_data['entry'] = np.zeros((T,), dtype=np.int32)
        if 'skill_id' not in stacked_data:
            stacked_data['skill_id'] = np.zeros((T,), dtype=np.int32)
        obs_dim = stacked_data['observations'].shape[1]
        if 'skill_aux' not in stacked_data:
            # In this design, skill_aux will later store the 512-d semantic embedding for each sample.
            stacked_data['skill_aux'] = np.zeros((T, self.semantic_dim), dtype=np.float32)
            for i in range(T):
                stacked_data['skill_aux'][i] = self.semantic_embeddings.get(
                    stacked_data['skills'][i], np.zeros(self.semantic_dim)
                )
            self.logger.info(f"skill_aux initialized with shape {stacked_data['skill_aux'].shape}")
        return dataloader

    def visualize_prototype(self, skill_id, proto_id, centroid_matrix):
        """
        Visualize a prototype matrix using an image plot.
        
        Parameters:
            skill_id: the integer skill id.
            proto_id: the unique id of the prototype.
            centroid_matrix: the prototype matrix (np.array) of shape (bases_num, obs_dim).
        """
        plt.figure()
        plt.imshow(centroid_matrix, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f"Prototype {proto_id} for Skill ID {skill_id}")
        plt.xlabel("Observation Dimension")
        plt.ylabel("Centroid Index")
        plt.tight_layout()
        # Save the plot to a file.
        plt.savefig(f"data/visualize/prototype_skill_{skill_id}_proto_{proto_id}.png")
        plt.close()

    def compute_cosine_sim(self, state_vec, matrix):
        """
        Compute cosine similarities between a state vector and each row in a matrix.
        
        Parameters:
            state_vec: a 1D array of shape (f,).
            matrix: a 2D array of shape (k, f), where each row is a centroid.
        
        Returns:
            sims: a 1D array of cosine similarities of length k.
        """
        norm_state = np.linalg.norm(state_vec)
        if norm_state == 0:
            norm_state = 1.0
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        sims = np.dot(matrix, state_vec) / (norm_state * norms)
        return sims

    def map_entry(self, dataloader):
        """
        1. Group data points based on the 'skills' field in the dataset.
        2. For each unique skill group, perform K-Means clustering on the observations.
        3. Create a new prototype (a centroid matrix of shape (bases_num, obs_dim)) for that skill.
           If the skill already exists, append the new prototype to its list.
        4. Update dataloader.stacked_data's 'entry' and 'skill_id' fields using the integer mapping for skills.
        5. Compute a new field 'decoder_id' for each sample by selecting the best matching prototype
           (using cosine similarity between the observation and each prototype's centroids).
           An error is raised if no prototype is available for a sample's skill.
        """
        stacked_data = dataloader.stacked_data
        if 'skills' not in stacked_data:
            raise ValueError("Stacked data must contain a 'skills' field with string labels")
        skills_arr = stacked_data['skills']  # shape: (T,), each element is a string.
        observations = stacked_data['observations']
        T = len(observations)

        # Process each unique skill.
        for skill in np.unique(skills_arr):
            indices = np.where(skills_arr == skill)[0]
            if len(indices) == 0:
                continue
            data_for_skill = observations[indices]
            
            # Perform K-Means clustering on the data for this skill.
            kmeans = KMeans(n_clusters=self.bases_num, random_state=0)
            kmeans.fit(data_for_skill)
            centroids = kmeans.cluster_centers_  # shape: (bases_num, obs_dim)

            # If this is the first occurrence of the skill, assign a new id.
            if skill not in self.skill_to_id:
                new_skill_id = len(self.skill_to_id)
                self.skill_to_id[skill] = new_skill_id
                self.entry_to_skill[new_skill_id] = skill
                self.skill_prototypes[new_skill_id] = []  # initialize empty list
                if self.config.verbose:
                    self.logger.info(f"New skill detected: '{skill}' assigned id {new_skill_id}")

            skill_id = self.skill_to_id[skill]
            # Create a new prototype (a tuple with unique id and centroid matrix)
            proto_id = self.next_prototype_id
            new_prototype = (proto_id, centroids)
            self.skill_prototypes[skill_id].append(new_prototype)
            if self.config.verbose:
                self.logger.info(f"Added prototype {proto_id} for skill '{skill}' (id {skill_id}) with shape {centroids.shape}")
            if self.config.visualize:
                self.visualize_prototype(skill_id, proto_id, centroids)
            self.next_prototype_id += 1

        # Update the 'entry' and 'skill_id' fields in the dataloader using the skill-to-id mapping.
        new_entry = np.zeros(T, dtype=np.int32)
        new_skill_id_field = np.zeros(T, dtype=np.int32)
        for i, skill in enumerate(skills_arr):
            id_val = self.skill_to_id.get(skill, -1)
            new_entry[i] = id_val
            new_skill_id_field[i] = id_val
        stacked_data['entry'] = new_entry
        stacked_data['skill_id'] = new_skill_id_field

        # Compute the 'decoder_id' for each sample based on the best matching prototype.
        decoder_ids = np.zeros(T, dtype=np.int32)
        for i in range(T):
            skill = stacked_data['skills'][i]
            if skill not in self.skill_to_id:
                raise ValueError(f"Skill '{skill}' not found in skill mapping")
            skill_id = self.skill_to_id[skill]
            proto_list = self.skill_prototypes.get(skill_id)
            if proto_list is None or len(proto_list) == 0:
                raise ValueError(f"No prototype available for skill '{skill}'")
            obs_vec = observations[i]  # Observation vector for sample i
            best_sim = -np.inf
            best_proto_id = -1
            # Loop over each prototype for the skill
            for proto in proto_list:
                proto_id, centroid_matrix = proto  # centroid_matrix shape: (bases_num, obs_dim)
                sims = self.compute_cosine_sim(obs_vec, centroid_matrix)
                proto_sim = np.max(sims)
                if proto_sim > best_sim:
                    best_sim = proto_sim
                    best_proto_id = proto_id
            # Assign the best prototype's id as the decoder id for sample i
            decoder_ids[i] = best_proto_id
        
        # Store the computed decoder_ids in stacked_data
        stacked_data['decoder_id'] = decoder_ids

        return dataloader

    def update_interface(self, dataloader):
        """
        Update the interface based on the dataloader data.
        The sequence is: init_entry -> map_entry -> update_dataloader.
        """
        self.init_entry(dataloader)
        self.map_entry(dataloader)
        return self.update_dataloader(dataloader)

    def update_dataloader(self, dataloader):
        """
        Mimics the PTGMInterface update: store the original observations,
        concatenate 'skill_aux' to 'observations'.
        """
        dataloader.stacked_data['orig_obs'] = dataloader.stacked_data['observations'].copy()
        dataloader.stacked_data['observations'] = np.concatenate(
            (dataloader.stacked_data['observations'], dataloader.stacked_data['skill_aux']), axis=-1
        )
        return dataloader

    def rollback_dataloader(self, dataloader):
        """
        Roll back the dataloader to its state before update_dataloader was called.
        """
        if 'orig_obs' not in dataloader.stacked_data:
            return dataloader
        dataloader.stacked_data['observations'] = dataloader.stacked_data['orig_obs']
        del dataloader.stacked_data['orig_obs']
        return dataloader

    def forward(self, entry, current_state):
        """
        For a given entry (skill id) and current state, compute the cosine similarity
        between the current state vector and each prototype's centroids for that skill.
        The best matching prototype is selected. Instead of returning its id as the auxiliary
        output, this version returns the 512-d semantic embedding corresponding to the skill
        associated with the input entry.
        
        Parameters:
            entry: A single value or an array (B,) of skill ids.
            current_state: A (B, f) array of current state vectors.
        
        Returns:
            A tuple: ((entry, chosen_proto_ids), semantic_output)
                where semantic_output is a (B, 512) array of semantic embeddings.
                chosen_proto_ids remains the selected prototype ids (for backward compatibility).
        
        Raises:
            ValueError if the skill id is not mapped or if the semantic embedding for the skill is not found.
        """
        current_state = np.array(current_state)
        if np.isscalar(entry):
            entry = [entry]
            current_state = current_state.reshape(1, -1)
        entry = np.array(entry, dtype=np.int32)
        batch_size = entry.shape[0]
        chosen_proto_ids = np.zeros(batch_size, dtype=np.int32)
        
        # Select the best prototype for each input entry using cosine similarity.
        for i in range(batch_size):
            skill_id = int(entry[i])
            if skill_id not in self.entry_to_skill:
                self.logger.warning(f"Skill id {skill_id} out of range, defaulting to 0")
                skill_id = 0
            proto_list = self.skill_prototypes.get(skill_id)

            if proto_list is None or len(proto_list) == 0:
                chosen_proto_ids[i] = 0  # No prototype available.
                continue
            else:
                state_vec = current_state[i]
                best_sim = -np.inf
                best_proto_id = -1
                for proto in proto_list:
                    proto_id, centroid_matrix = proto  # centroid_matrix shape: (k, f)
                    sims = self.compute_cosine_sim(state_vec, centroid_matrix)
                    proto_sim = np.max(sims)
                    if proto_sim > best_sim:
                        best_sim = proto_sim
                        best_proto_id = proto_id
                chosen_proto_ids[i] = best_proto_id

        # For each input entry, retrieve the corresponding 512-d semantic embedding using the skill name.
        semantic_output = []
        for i in range(batch_size):
            skill_id = int(entry[i])
            if skill_id not in self.entry_to_skill:
                self.logger.warning(f"Skill id {skill_id} out of range, defaulting to 0")
                skill_id = 0
            skill_name = self.entry_to_skill[skill_id]
            if skill_name not in self.semantic_embeddings:
                raise ValueError(f"Semantic embedding for skill '{skill_name}' not found")
            semantic_output.append(self.semantic_embeddings[skill_name])
        semantic_output = np.array(semantic_output)
        
        # NOTE: For IsCiL, as it uses a unique decoder per prototype, the decoder id is the same as the prototype id.
        # Return a tuple where the first element is a tuple of (entry, chosen_proto_ids) and the second element is the chosen semantic_output.
        return (entry, chosen_proto_ids), semantic_output

    # -------------------------
    # skill mapping method
    # -------------------------
    def get_skill_ids(self, skills) :
        """
        Given a list of skills, return their corresponding skill ids.
        
        Parameters:
            skills: A list of skill strings.
        
        Returns:
            A list of skill ids corresponding to the input skills.
        """
        skill_ids = []
        for skill in skills:
            if skill in self.skill_to_id:
                skill_ids.append(self.skill_to_id[skill])
            else:
                skill_ids.append(0) # unknown skill to 0(default)
                # raise ValueError(f"Skill '{skill}' not found in skill mapping")
        return np.array(skill_ids)
# ---------------------------------------------------------------------
# Enhanced Test Function for PrototypeInterface
# ---------------------------------------------------------------------
def test_semantic_interface():
    """
    This test function verifies the functionality of SemanticInterface by processing
    multiple dataset chunks (using BaseDataloader and DEFAULT_DATASTREAM). It checks that:
      1. The interface correctly maps skills from the 'skills' field to fixed integer entries.
      2. The internal skill mapping is correctly maintained.
      3. The forward method returns the fixed entries as both the skill id and the auxiliary output.
      4. The dataloader update and rollback functions work as expected.
    """
    import numpy as np
    from SILGym.dataset.dataloader import BaseDataloader
    from SILGym.config.skill_stream_config import DEFAULT_DATASTREAM
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Define a simple configuration class
    class Config:
        def __init__(self, verbose=False):
            self.verbose = verbose

    config = Config(verbose=True)

    interface = SemanticInterface(config)

    # Process first two dataset chunks if available.
    num_chunks = len(DEFAULT_DATASTREAM)
    for idx in range(num_chunks):
        console.rule(f"[bold red]Processing dataset chunk {idx}[/bold red]")
        # Load dataset chunk using BaseDataloader.
        dataloader = BaseDataloader(data_paths=DEFAULT_DATASTREAM[idx].dataset_paths)

        # If the 'skills' field is not present, add a dummy skills field.
        if 'skills' not in dataloader.stacked_data:
            T = len(dataloader.stacked_data['observations'])
            # For testing, assign "skillA" for indices divisible by 2, and "skillB" for the rest.
            skills = np.array(["skillA" if i % 2 == 0 else "skillB" for i in range(T)])
            dataloader.stacked_data['skills'] = skills

        # Update the interface with the current dataloader.
        dataloader = interface.update_interface(dataloader)

        # Display internal skill mappings.
        table = Table(title="Internal Skill Mappings")
        table.add_column("Skill", style="cyan")
        table.add_column("Skill ID", style="magenta")
        for skill, skill_id in interface.skill_to_id.items():
            table.add_row(skill, str(skill_id))
        console.print(table)

        # Prepare test entries from the dataloader's 'entry' field.
        unique_skill_ids = np.unique(dataloader.stacked_data['entry'])
        test_indices = []
        for skill in unique_skill_ids:
            idxs = np.where(dataloader.stacked_data['entry'] == skill)[0]
            if len(idxs) > 0:
                test_indices.append(idxs[0])
        test_indices = np.array(test_indices)
        test_entries = dataloader.stacked_data['entry'][test_indices]
        
        # Display test entries.
        table_test = Table(title="Test Entries (Skill IDs)")
        table_test.add_column("Index", justify="center")
        table_test.add_column("Skill ID", style="cyan", justify="center")
        for i, idx in enumerate(test_indices):
            table_test.add_row(str(idx), str(dataloader.stacked_data['entry'][idx]))
        console.print(table_test)

        # --- Test the forward method ---
        console.print("\n[bold green]Testing forward method[/bold green]")
        for i, entry in enumerate(test_entries):
            # For SemanticInterface, the forward pass simply returns the entry as both outputs.
            out_skill_id, _ = interface.forward(entry)
            _ , aux = out_skill_id
            console.print(f"Test entry {i}: input skill ID {entry}, forward output: skill id = {out_skill_id}, aux = {aux}")

        # Update and then rollback the dataloader.
        dataloader = interface.update_dataloader(dataloader)
        dataloader = interface.rollback_dataloader(dataloader)
        console.print(f"[bold green]After update and rollback, observations shape: {dataloader.stacked_data['observations'].shape}")
        console.print("--------------------------------------------------", style="bold")

def test_prototype_interface():
    """
    This test function verifies the functionality of PrototypeInterface by processing
    multiple dataset chunks (using BaseDataloader and DEFAULT_DATASTREAM). It checks that:
      1. The interface correctly groups observations by the 'skills' field.
      2. For each skill, a list of prototypes is maintained (each prototype is a tuple of (prototype_id, centroid_matrix)).
      3. When processing multiple chunks, new skills and new prototypes are added.
      4. The forward method computes cosine similarities between the current state and each prototype's centroids,
         and returns the prototype id with the highest similarity.
      5. The dataloader can be rolled back to its original state.
      6. **Additional**: Evaluate using all dataset chunks and print the unique prototype ids returned by the forward method.
    """
    import numpy as np
    from SILGym.dataset.dataloader import BaseDataloader
    from SILGym.config.skill_stream_config import DEFAULT_DATASTREAM
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Create a configuration instance with bases_num set to 5, with verbose logging and visualization enabled.
    config = IsCiLInterfaceConfig(bases_num=20, verbose=True, visualize=True)
    interface = PrototypeInterface(config)

    # Process all dataset chunks in DEFAULT_DATASTREAM.
    num_chunks = len(DEFAULT_DATASTREAM)
    
    # List to store all prototype ids returned from the forward method across all chunks.
    all_forward_proto_ids = []

    for idx in range(num_chunks):
        console.rule(f"[bold red]Processing dataset chunk {idx}[/bold red]")
        # Load the dataset chunk using BaseDataloader.
        dataloader = BaseDataloader(data_paths=DEFAULT_DATASTREAM[idx].dataset_paths)

        # If the 'skills' field is not present, add dummy data.
        if 'skills' not in dataloader.stacked_data:
            T = len(dataloader.stacked_data['observations'])
            if len(dataloader.stacked_data['observations'].shape) > 1:
                f = dataloader.stacked_data['observations'].shape[1]
            else:
                f = 1
            # For example, assign "skillA" for indices divisible by 3, and "skillB" for the rest.
            skills = np.array(["skillA" if i % 3 == 0 else "skillB" for i in range(T)])
            dataloader.stacked_data['skills'] = skills

        # Update the interface (initialize entries, map skills, and update dataloader).
        dataloader = interface.update_interface(dataloader)

        console.print(f"[bold green]After update_interface:[/bold green] Augmented observations shape: {dataloader.stacked_data['observations'].shape}")
        console.print(f"Original observations (orig_obs) shape: {dataloader.stacked_data['orig_obs'].shape}")

        # Display internal skill mapping information.
        table = Table(title="Internal Skill Mappings")
        table.add_column("Skill", style="cyan")
        table.add_column("Skill ID", style="magenta")
        table.add_column("No. of Prototypes", justify="right", style="green")
        for skill, skill_id in interface.skill_to_id.items():
            proto_list = interface.skill_prototypes.get(skill_id, [])
            table.add_row(skill, str(skill_id), str(len(proto_list)))
        console.print(table)

        # Display details of prototypes for each skill.
        for skill_id, proto_list in interface.skill_prototypes.items():
            table_proto = Table(title=f"Prototypes for Skill ID {skill_id} ({interface.entry_to_skill[skill_id]})")
            table_proto.add_column("Prototype ID", style="yellow")
            table_proto.add_column("Matrix Shape", style="blue")
            for proto in proto_list:
                proto_id, matrix = proto
                table_proto.add_row(str(proto_id), str(matrix.shape))
            console.print(table_proto)
        console.print("--------------------------------------------------", style="bold")

        # --- Prepare test entries by selecting the first index for each unique skill ---
        unique_skill_ids = np.unique(dataloader.stacked_data['entry'])
        test_indices = []
        for skill in unique_skill_ids:
            idxs = np.where(dataloader.stacked_data['entry'] == skill)[0]
            if len(idxs) > 0:
                test_indices.append(idxs[0])
        test_indices = np.array(test_indices)
        test_entries = dataloader.stacked_data['entry'][test_indices]
        orig_obs = dataloader.stacked_data['orig_obs']
        # Use the corresponding rows from orig_obs as the current state.
        current_state = orig_obs[test_indices]

        table_test = Table(title="Test Entries (from multiple skills)")
        table_test.add_column("Index", justify="center")
        table_test.add_column("Skill ID", style="cyan", justify="center")
        table_test.add_column("Skill", style="magenta", justify="center")
        for i, idx_val in enumerate(test_indices):
            skill_id = dataloader.stacked_data['entry'][idx_val]
            skill_str = interface.entry_to_skill.get(skill_id, "Unknown")
            table_test.add_row(str(idx_val), str(skill_id), skill_str)
        console.print(table_test)

        # --- Test the forward method ---
        console.print("\n[bold green]Testing forward method[/bold green]")
        table_state = Table(title="Current State Vectors Shapes")
        table_state.add_column("Test Index", justify="center")
        table_state.add_column("Shape", justify="center")
        for i, vec in enumerate(current_state):
            table_state.add_row(str(i), str(vec.shape))
        console.print(table_state)

        # For each test entry, calculate cosine similarities with each prototype.
        for i, skill_id in enumerate(test_entries):
            proto_list = interface.skill_prototypes.get(skill_id)
            state_vec = current_state[i]
            if proto_list is None or len(proto_list) == 0:
                console.print(f"[red]No prototype found for skill ID {skill_id}[/red]")
            else:
                best_overall_sim = -np.inf
                best_proto_id = -1
                console.print(f"\n[bold blue]Processing test entry index {i} with skill ID {skill_id} ({interface.entry_to_skill.get(skill_id, 'Unknown')})[/bold blue]")
                for proto in proto_list:
                    proto_id, matrix = proto
                    sims = interface.compute_cosine_sim(state_vec, matrix)
                    best_sim = np.max(sims)
                    console.print(f"  Prototype ID {proto_id} similarities: {sims}, best: {best_sim:.4f}")
                    if best_sim > best_overall_sim:
                        best_overall_sim = best_sim
                        best_proto_id = proto_id
                console.print(f"  [bold]Chosen prototype id for test entry index {i}: {best_proto_id} with similarity {best_overall_sim:.4f}[/bold]")

        # Call the forward method from the interface.
        skill_ids, _ = interface.forward(test_entries, current_state)
        table_forward = Table(title="Forward Method Output")
        table_forward.add_column("Test Skill ID", style="cyan")
        table_forward.add_column("Chosen Prototype ID", style="magenta")
        for sid, pid in zip(skill_ids[0], skill_ids[1]):
            table_forward.add_row(str(sid), str(pid))
        console.print(table_forward)
        
        # Append the forward method's prototype ids to the list.
        all_forward_proto_ids.extend(skill_ids[1].tolist())

        # Rollback the dataloader to its original state.
        dataloader = interface.rollback_dataloader(dataloader)
        console.print(f"[bold green]After rollback_dataloader:[/bold green] Restored observations shape: {dataloader.stacked_data['observations'].shape}")
        console.print("--------------------------------------------------", style="bold")

    # --------------------------
    # Print unique prototype ids from the forward method across all dataset chunks.
    # --------------------------
    if all_forward_proto_ids:
        unique_proto_ids = np.unique(np.array(all_forward_proto_ids))
        table_unique = Table(title="Unique Prototype IDs Across All Dataset Chunks")
        table_unique.add_column("Prototype ID", style="magenta", justify="center")
        for proto_id in unique_proto_ids:
            table_unique.add_row(str(proto_id))
        console.print(table_unique)
    else:
        console.print("[red]No prototype ids were returned from the forward method.[/red]")

if __name__ == '__main__':
    test_semantic_interface()
    # test_prototype_interface()
