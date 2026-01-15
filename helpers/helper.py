import torch
import numpy as np
import pandas as pd
import copy
import requests
import gzip
import os
import ssl
import urllib.request
from typing import Dict, Tuple, List
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import random
from tqdm import tqdm

import time
from functools import wraps
from multiprocessing import Pool


def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High-precision timer
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result

    return wrapper


UNIPROT_FASTA_URL = "https://www.uniprot.org/uniprot/{uid}.fasta"
UNIPROT_SEARCH_URL = "https://www.uniprot.org/uniprot/"

SCOP_URL = "https://scop.berkeley.edu/downloads/scopeseq-2.07/astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa"
FILENAME = "astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa"

# Exclusion Lists for Experiment 3 (from the paper)
ROSSMANN_EXCLUSIONS = ["c.2", "c.3", "c.4", "c.5", "c.27", "c.28", "c.30", "c.31"]
BETA_PROP_EXCLUSIONS = ["b.66", "b.67", "b.68", "b.69", "b.70"]


def randomize_model(model):
    # 1. Move original to CPU temporarily to avoid GPU-copy errors
    device = next(model.parameters()).device
    model.cpu()

    # 2. Create the copy safely on CPU
    untrained_model = copy.deepcopy(model)

    # 3. Move original back to its device immediately
    model.to(device)

    # 4. Randomize weights on the copy
    for name, param in untrained_model.named_parameters():
        if "weight" in name and param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        elif "bias" in name:
            torch.nn.init.constant_(param, 0)

    # 5. Return untrained model (caller handles moving it to GPU)
    return untrained_model


@time_execution
def get_hidden_representations(model, alphabet, labels, sequences, batch_size=1):
    """
    Verarbeitet Sequenzen in Batches durch das ESM-Modell.
    Nutzt multiprocessing, um die Tokenisierung zu beschleunigen.

    Args:
        model: Das geladene ESM Modell.
        alphabet: Das zum Modell gehörige Alphabet-Objekt.
        labels: Liste von Strings (Namen der Proteine).
        sequences: Liste von Strings (Aminosäuresequenzen).
        batch_size: Anzahl der Sequenzen pro Durchlauf (Integer).

    Returns:
        tuple: (all_token_representations [List of Tensors], all_strs [List])
    """
    # 1. Gerät bestimmen und Batch Converter vorbereiten
    device = next(model.parameters()).device
    batch_converter = alphabet.get_batch_converter()

    # Daten zippen
    data = list(zip(labels, sequences))
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    all_token_representations = []
    all_strs = []

    # 2. Layer identifizieren (Output Layer)
    repr_layer = model.num_layers

    print(
        f"Processing {len(data)} sequences in batches of {batch_size} using multiprocessing..."
    )

    # Use a multiprocessing Pool to parallelize tokenization
    # We use imap for lazy evaluation to save memory
    with Pool() as pool:
        # pool.imap is lazy and will process batches in parallel
        tokenized_batches_iterator = pool.imap(batch_converter, batches)

        # 3. Schleife über die tokenisierten Daten
        for batch_labels, batch_strs, batch_tokens in tqdm(
            tokenized_batches_iterator, total=len(batches)
        ):
            batch_tokens = batch_tokens.to(device)

            # 4. Inferenz
            with torch.no_grad():
                results = model(
                    batch_tokens, repr_layers=[repr_layer], return_contacts=False
                )

            token_representations = results["representations"][repr_layer]

            # 5. Ergebnisse speichern
            for j, tokens in enumerate(batch_tokens):
                seq_len = len(batch_strs[j]) + 2  # +2 for <cls> and <eos>
                seq_rep = token_representations[j, :seq_len].cpu()
                all_token_representations.append(seq_rep)

            all_strs.extend(batch_strs)
            torch.cuda.empty_cache()

    return all_token_representations, all_strs


@time_execution
def get_protein_embedding(token_representations, batch_strs):
    """
    Führt Mean Pooling auf den Hidden Representations durch, um ein Embedding pro Protein zu erhalten.
    TODO: Mean pooling: exclude special tokens & padding.

    Args:
        token_representations (List[Tensor]): Output von get_hidden_representations (Liste von Tensoren).
        batch_strs (List[str]): Liste der Sequenzen.

    Returns:
        numpy.ndarray: Array der Embeddings (Shape: [N, hidden_dim]).
    """
    embeddings = []

    # Iteration durch den Batch
    for i, seq_str in enumerate(batch_strs):
        seq_len = len(seq_str)

        # FIX: Zugriff auf Liste zuerst mit [i], dann Slicing des Tensors mit [1 : ...]
        # Wir nehmen Index 1 bis seq_len + 1, um Start-Token (<cls>) und End-Token (<eos>) zu ignorieren.
        # Dies entspricht dem "averaging across the hidden representation at each position".

        # token_representations[i] ist der Tensor für das i-te Protein.
        seq_tensor = token_representations[i]

        # Slicing: [1 : seq_len + 1] extrahiert nur die Aminosäuren.
        # .mean(0) berechnet den Durchschnitt über die Länge (Dimension 0).
        seq_embedding = seq_tensor[1 : seq_len + 1].mean(0)

        embeddings.append(seq_embedding.cpu().numpy())

    return np.array(embeddings)


# Make sure these constants are defined correctly at the top of your script
DATA_DIR = "data"
SCOP_FULL_PATH = os.path.join(DATA_DIR, FILENAME)

def download_no_requests():
    """Downloads using standard urllib with SSL verification disabled to the data directory."""
    
    # 1. Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # 2. Check if the file already exists at the FULL path
    if os.path.exists(SCOP_FULL_PATH):
        print(f"File already exists at {SCOP_FULL_PATH}. Skipping download.")
        return

    print(f"Downloading {FILENAME} to {DATA_DIR} using urllib (SSL ignored)...")

    # Create an unverified SSL context to bypass the error
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with (
            urllib.request.urlopen(SCOP_URL, context=ctx) as response,
            open(SCOP_FULL_PATH, "wb") as out_file, # <--- Changed to write to full path
        ):
            out_file.write(response.read())
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please try the Manual Download option below.")


def parse_and_filter_scop():
    """Parses FASTA headers and filters out excluded folds."""
    data = []

    # 1. Check existence and try to download if missing
    if not os.path.exists(SCOP_FULL_PATH):
        print(f"File not found at {SCOP_FULL_PATH}. Attempting download...")
        download_no_requests()

    # 2. Final check
    if not os.path.exists(SCOP_FULL_PATH):
        print("Error: File could not be found or downloaded.")
        return pd.DataFrame()

    print(f"Parsing and filtering sequences from {SCOP_FULL_PATH}...")
    with open(SCOP_FULL_PATH, "r") as f:
        current_header = None
        current_seq = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    # --- THE FIX: Call process_entry to filter exclusions ---
                    process_entry(current_header, "".join(current_seq), data)
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)

        if current_header:
             # --- Don't forget the last entry! ---
            process_entry(current_header, "".join(current_seq), data)

    return pd.DataFrame(data)


def process_entry(header, sequence, data_list):
    # Header format example: >d1dlwa_ a.4.5.1 (A:)
    parts = header.split()
    if len(parts) < 2:
        return

    # print(parts)

    domain_id = parts[0][1:]
    scop_code = parts[1]
    hierarchy = scop_code.split(".")

    if len(hierarchy) < 4:
        return

    fold = f"{hierarchy[0]}.{hierarchy[1]}"
    superfamily = f"{hierarchy[0]}.{hierarchy[1]}.{hierarchy[2]}"
    family = scop_code

    # FILTERING LOGIC (Phase 1 Rules)
    # New (Correct):
    is_rossmann = fold in ROSSMANN_EXCLUSIONS
    is_beta_prop = fold in BETA_PROP_EXCLUSIONS

    if not is_rossmann and not is_beta_prop:
        data_list.append(
            {
                "domain_id": domain_id,
                "sequence": sequence,
                "class": hierarchy[0],
                "fold": fold,
                "superfamily": superfamily,
                "family": family,
            }
        )


def get_pfam_seed_by_id(pfam_accession):
    """
    Streams the Pfam-A seed database from EBI FTP and extracts a specific family.
    Bypasses broken API endpoints.
    """
    # Official EBI FTP URL for the current release of Pfam Seed Alignments
    url = "http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz"

    print(f"Streaming Pfam database to find {pfam_accession}...")
    print("This may take 1-2 minutes as it searches the compressed stream.")

    with urllib.request.urlopen(url) as response:
        # Decompress the stream on the fly
        with gzip.open(response, "rt") as f:
            entry_buffer = []
            for line in f:
                entry_buffer.append(line)
                # End of a record in Stockholm format
                if line.startswith("//"):
                    block = "".join(entry_buffer)
                    # Check if this block matches our target Accession
                    if f"AC   {pfam_accession}" in block:
                        return block
                    entry_buffer = []  # Reset buffer for next entry
    return None


def plot_tsne_exp1(amino_acids_df, representations):
    amino_acids_df = amino_acids_df.copy()
    property_map = {
        "Hydrophobic": ("green", "o"),
        "Aromatic": ("green", "+"),
        "Polar": ("blue", "o"),
        "Unique": ("orange", "o"),
        "Negatively charged": ("red", "x"),
        "Positively charged": ("red", "s"),
    }

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    emb = tsne.fit_transform(representations)
    amino_acids_df["TSNE-1"] = emb[:, 0]
    amino_acids_df["TSNE-2"] = emb[:, 1]
    amino_acids_df
    fig, ax = plt.subplots(figsize=(10, 6))
    for property_name, (color, marker) in property_map.items():
        mask = amino_acids_df["property"].str.capitalize() == property_name
        ax.scatter(
            amino_acids_df[mask]["TSNE-1"],
            amino_acids_df[mask]["TSNE-2"],
            c=color,
            marker=marker,
            label=property_name,
            s=100,
        )

    # Annotate points with amino acid labels
    for i, row in amino_acids_df.iterrows():
        plt.annotate(row["Amino Acid"], (row["TSNE-1"] + 2, row["TSNE-2"]), fontsize=8)

    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.legend()
    plt.show()


def sample_aligned_pairs(
    mapping_list: List[Dict],
    num_samples=50000,
    msa: List[str] = None,
    mode: str = "all",
) -> List[Tuple[int, int, int, int]]:
    """
    Samples aligned residue pairs from a list of mappings.

    Args:
        mapping_list (List[Dict]): List of mappings with 'query_pos' and 'template_pos'.
        num_seqs (int): Number of sequences.
        num_samples (int): The number of pairs to sample.
        msa (List[str], optional): List of sequence strings. Required for 'identity' or 'distinct' modes.
        mode (str, optional): 'all', 'identity', or 'distinct'. Defaults to 'all'.
    """
    num_seqs = len(msa)
    print(mode)
    if mode not in ["all", "identity", "distinct"]:
        raise ValueError("mode must be one of 'all', 'identity', or 'distinct'")

    if mode != "all":
        if msa is None:
            raise ValueError("msa must be provided for 'identity' or 'distinct' mode.")
        if len(msa) != num_seqs:
            raise ValueError("length of msa must be equal to num_seqs")

    aligned_pairs = []
    # --- Sampling Aligned Pairs ---
    # Pick a random column in the MSA, find two sequences that both have a residue there.
    attempts = 0
    while len(aligned_pairs) < num_samples and attempts < num_samples * 5:
        attempts += 1

        # Pick two random sequences
        seq_a_idx = random.randint(0, num_seqs - 1)
        seq_b_idx = random.randint(0, num_seqs - 1)

        if seq_a_idx == seq_b_idx:
            continue

        # Check if both sequences have a residue at this MSA column (no gap)
        map_a = mapping_list[seq_a_idx]
        map_b = mapping_list[seq_b_idx]

        alligned_col = random.choice(list(map_a.keys()))
        alligned_col = int(alligned_col)

        if alligned_col in map_a.keys() and alligned_col in map_b.keys():
            if mode != "all":
                aa_a = msa[seq_a_idx][alligned_col]
                aa_b = msa[seq_b_idx][alligned_col]

                if mode == "identity":
                    if aa_a != aa_b:
                        continue
                elif mode == "distinct":
                    if aa_a == aa_b:
                        continue
            # Get the embeddings for these specific residues
            # Note: token_reps includes <cls> at 0, so we add 1 to the mapped index
            aligned_pairs.append((seq_a_idx, seq_b_idx, alligned_col, alligned_col))
    return aligned_pairs


def sample_unaligned_pairs(
    aligned_pairs: List[Tuple[int, int, int, int]],
    mapping_list: List[Dict],
) -> List[Tuple[int, int, int, int]]:
    """
    Samples unaligned residue pairs from a list of mappings and aligned pairs for the background distribution.
    Take into account possible bias because of positional encodings.

    Args:
        mapping_list (List[Dict]): List of mappings with 'query_pos' and 'template_pos'.
        msa_length (int): Length of the MSA.
    """
    unaligned_indices = []
    for seq_a_idx, seq_b_idx, aligned_col, _ in aligned_pairs:
        # 1. Calculate the absolute value of the difference of each residue's within-sequence positions
        map_a = mapping_list[seq_a_idx]
        map_b = mapping_list[seq_b_idx]
        pos_diff = abs(map_a[aligned_col] - map_b[aligned_col])

        attempts_unaligned = 0
        max_attempts = 50000
        while attempts_unaligned < max_attempts:
            attempts_unaligned += 1
            # 2. Select a pair of sequences at random
            seq_c_idx = random.choice(list(range(len(mapping_list))))
            seq_d_idx = random.choice(list(range(len(mapping_list))))

            if seq_c_idx == seq_d_idx:
                continue

            # 3. For that pair of sequences, select a pair of residues at random
            # whose absolute value of positional difference equals the one calculated above
            map_c = mapping_list[seq_c_idx]
            map_d = mapping_list[seq_d_idx]
            reversed_map_d = {v: k for k, v in map_d.items()}

            pos_c = int(random.choice(list(map_c.keys())))
            real_pos_c = map_c[pos_c]

            sign = random.choice([-1, 1])
            real_pos_d = real_pos_c + sign * pos_diff
            if real_pos_d < 0:
                real_pos_d = real_pos_c - sign * pos_diff

            if real_pos_d in reversed_map_d:
                pos_d = reversed_map_d[real_pos_d]

                # 4. Verify that the residues are unaligned (different MSA columns)
                if pos_c != pos_d:
                    unaligned_indices.append((seq_c_idx, seq_d_idx, pos_c, pos_d))
                    break

    return unaligned_indices


def get_cosine_similarity(
    seq_pairs: List[Tuple[int, int, int, int]], token_reps, mapping_list, msa=None
):
    """
    Computes cosine similarity for given sequence pairs using vectorized operations.

    Args:
        seq_pairs (List[Tuple[int, int, int]]): List of tuples (seq_a_idx, seq_b_idx, aligned_col).

    """
    # Extract embeddings for all pairs at once
    emb_a_list = []
    emb_b_list = []

    for seq_a_idx, seq_b_idx, col_a_idx, col_b_idx in seq_pairs:
        map_a = mapping_list[seq_a_idx]
        map_b = mapping_list[seq_b_idx]

        # check alignment identity
        # if msa is not None:
        #     print(msa[seq_a_idx][col_a_idx] == msa[seq_b_idx][col_b_idx])

        # +1 because of <cls> token at position 0
        emb_a_list.append(token_reps[seq_a_idx][map_a[col_a_idx] + 1].cpu().numpy())
        emb_b_list.append(token_reps[seq_b_idx][map_b[col_b_idx] + 1].cpu().numpy())

    # Stack into matrices: shape [num_pairs, embedding_dim]

    emb_a = np.array(emb_a_list)
    emb_b = np.array(emb_b_list)

    # Compute all similarities at once
    sims = np.sum(emb_a * emb_b, axis=1) / (
        np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1)
    )

    # sims = cosine_similarity(emb_a, emb_b).diagonal()

    return sims.tolist()


def clean_and_map_msa(msa):
    clean_seqs = []
    msa_mapping = []  # List of dictionaries: {msa_column_index: sequence_residue_index}

    print(f"Processing {len(msa)} sequences...")

    for record in msa:
        original_seq = str(record.seq).upper()
        clean_seq = ""
        mapping = {}

        seq_idx = 0
        for msa_idx, char in enumerate(original_seq):
            # Pfam uses '.' and '-' for gaps. We only want amino acids.
            if char not in [".", "-"]:
                clean_seq += char
                # Record that this MSA column corresponds to this index in the clean sequence
                mapping[msa_idx] = seq_idx
                seq_idx += 1

        clean_seqs.append(clean_seq)
        msa_mapping.append(mapping)
    return clean_seqs, msa_mapping


def forward_pass(clean_seqs, model, alphabet):
    # print("Running forward pass on TRAINED model...")
    # Note: We use the helper, but we pass the labels=None explicitly if your helper expects it
    batch_size = 32
    with torch.no_grad():
        token_reps_trained, batch_strs_trained = get_hidden_representations(
            model,
            alphabet,
            np.zeros_like(clean_seqs),
            clean_seqs,
            batch_size=batch_size,
        )

    # B. Untrained Model (Random Baseline)
    # print("Running forward pass on UNTRAINED model...")
    # Create a randomized version of the model to test the "Emergence" hypothesis
    untrained_model = randomize_model(model)
    if torch.cuda.is_available():
        untrained_model = untrained_model.cuda()

    with torch.no_grad():
        token_reps_untrained, batch_strs_untrained = get_hidden_representations(
            untrained_model, alphabet, np.zeros_like(clean_seqs), clean_seqs, batch_size
        )
    return token_reps_trained, token_reps_untrained
    # --- Evaluation (AUC) ---


# Quantify the separation between the distributions
def get_auc(
    train_aligned_sim,
    train_unaligned_sim,
    untrain_aligned_sim,
    untrain_unaligned_sim,
    print_stats=False,
):
    def calculate_auc(aligned, unaligned):
        y_true = [1] * len(aligned) + [0] * len(unaligned)
        y_scores = np.concatenate([aligned, unaligned])
        return roc_auc_score(y_true, y_scores)

    auc_trained = calculate_auc(train_aligned_sim, train_unaligned_sim)
    auc_untrained = calculate_auc(untrain_aligned_sim, untrain_unaligned_sim)

    if print_stats:
        print(f"AUC Score (Discrimination between Aligned/Unaligned):")
        print(f"Trained Model:   {auc_trained:.4f}")
        print(f"Untrained Model: {auc_untrained:.4f}")
    return auc_trained, auc_untrained


# Plot Helper
def plot_dist(ax, aligned, unaligned, title):
    # Plot histograms with density
    ax.hist(
        aligned,
        bins=50,
        density=True,
        alpha=0.6,
        color="darkblue",
        label="Aligned pairs",
    )
    ax.hist(
        unaligned,
        bins=50,
        density=True,
        alpha=0.6,
        color="lightblue",
        label="Unaligned pairs",
    )
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=14)
    ax.set_xlim(0.5, 1)


def fetch_orthologous_groups(groups, sample_size=None):
    """Fetches sequence data for given COG groups from EggNOG API."""
    data = []
    print(f"Fetching data for {len(groups)} groups...")
    
    for group in groups:
        url = f"http://eggnogapi5.embl.de/nog_data/json/fasta/{group}"
        try:
            response = requests.get(url)
            if response.status_code != 200: continue
            
            fasta_content = response.json().get('fasta', '')
            for item in fasta_content.split('>')[1:]:
                header = item.splitlines()[0]
                seq = item.splitlines()[1]
                
                # Filter out ambiguous/long sequences
                if len(seq) < 1024 and 'J' not in seq:
                    data.append({
                        "group_id": group,
                        "specimen": header.split('.')[0],
                        "sequence": seq
                    })
        except Exception as e:
            print(f"Error fetching {group}: {e}")
        time.sleep(0.1)
        
    df = pd.DataFrame(data)
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df

def get_unigram_embeddings(sequences):
    """Calculates amino acid frequency vectors (baseline)."""
    vocab = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(vocab)}
    embeddings = np.zeros((len(sequences), len(vocab)))
    
    for i, seq in enumerate(sequences):
        if len(seq) == 0: continue
        for char in seq.upper():
            if char in aa_to_idx:
                embeddings[i, aa_to_idx[char]] += 1
        embeddings[i] /= len(seq)
    return embeddings



def get_phylum_mock(species_id):
    """
    Mock function to map species ID to phylum. 
    In a real scenario, use NCBI Taxonomy. 
    Here we map common model organisms and randomize others to simulate diversity.
    """
    # Map common IDs to real phyla
    known = {
        "9606": "Chordata", "10090": "Chordata", "7227": "Arthropoda", 
        "6239": "Nematoda", "3702": "Streptophyta", "4932": "Ascomycota",
        "562": "Proteobacteria", "511145": "Proteobacteria"
    }
    return known.get(str(species_id), f"Phylum_{str(species_id)[-1]}")

def greedy_cog_selection(df, target_k=25):
    """
    Selects k COGs maximizing intersecting species count.
    """
    # 1. Map each group to its set of species
    cog_species_map = df.groupby('group_id')['specimen'].apply(set).to_dict()
    
    # 2. Sort candidates by species count (descending)
    candidates = sorted(cog_species_map.keys(), key=lambda x: len(cog_species_map[x]), reverse=True)
    
    selected_cogs = []
    # Start with the largest group
    current_intersection = cog_species_map[candidates[0]]
    selected_cogs.append(candidates[0])
    
    # 3. Greedily add groups that maintain the largest intersection
    # (Simplified for demo: we just pick top K that have *any* overlap with current set)
    for cog in candidates[1:]:
        if len(selected_cogs) >= target_k:
            break
        
        # Check if adding this COG keeps the intersection reasonable (e.g., > 5 species)
        temp_intersection = current_intersection.intersection(cog_species_map[cog])
        if len(temp_intersection) > 4: 
            selected_cogs.append(cog)
            current_intersection = temp_intersection
            
    return selected_cogs, current_intersection

def sample_diverse_species(species_list, target_count=24):
    """
    Selects one species per phylum to ensure diversity[cite: 920].
    """
    phylum_map = {}
    for sp in species_list:
        ph = get_phylum_mock(sp)
        if ph not in phylum_map:
            phylum_map[ph] = []
        phylum_map[ph].append(sp)
        
    selected_species = []
    # Pick one from each phylum
    for ph in list(phylum_map.keys())[:target_count]:
        selected_species.append(phylum_map[ph][0])
        
    return selected_species