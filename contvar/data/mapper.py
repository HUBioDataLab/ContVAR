import os
import glob
import json
import random


class TripletDataPathMapper:
    """Maps protein file structure to anchor-positive-negative triplets.

    New split logic (per-family hold-out):
        - ALL 97 protein families participate in BOTH training and validation.
        - For each family, 2 positives + 2 negatives are held out for validation.
        - The remaining variants (typically 48+48) are used for training.
        - No test split is created.
        - The split can be saved to / loaded from a JSON file for reproducibility.
    """

    SPLIT_FILENAME = "split.json"

    def __init__(self, root_dir, val_pos=2, val_neg=2, seed=42, split_path=None):
        """
        Args:
            root_dir: Path to protein_triplets_data directory.
            val_pos: Number of positive variants to hold out per family for validation.
            val_neg: Number of negative variants to hold out per family for validation.
            seed: Random seed for reproducible split generation.
            split_path: If provided, load an existing split from this JSON file
                        instead of generating a new one. Set to None to create fresh.
        """
        self.root_dir = root_dir
        self.val_pos = val_pos
        self.val_neg = val_neg
        self.seed = seed

        # All protein family data (full, before splitting)
        self.triplets = []

        # Per-family split: train and val variants separated
        self.train_triplets = []
        self.val_triplets = []

        self._map_data()

        if split_path and os.path.exists(split_path):
            self._load_split(split_path)
        else:
            self._split_data()

    def _map_data(self):
        """Discover all protein families and their variant files."""
        originals = glob.glob(os.path.join(self.root_dir, 'originals', "*.cif"))

        for anchor in originals:
            prot_id = os.path.splitext(os.path.basename(anchor))[0]
            pos_dir = os.path.join(self.root_dir, 'positives', prot_id)
            neg_dir = os.path.join(self.root_dir, 'negatives', prot_id)

            p_files = sorted(glob.glob(os.path.join(pos_dir, "*.cif")))
            n_files = sorted(glob.glob(os.path.join(neg_dir, "*.cif")))

            if p_files and n_files:
                self.triplets.append({
                    'anchor': anchor,
                    'positives': p_files,
                    'negatives': n_files,
                    'protein_id': prot_id
                })

        # Sort by protein_id for deterministic ordering
        self.triplets.sort(key=lambda t: t['protein_id'])
        print(f"Found {len(self.triplets)} protein families")

    def _split_data(self):
        """Hold out val_pos positives + val_neg negatives per family."""
        random.seed(self.seed)
        self.train_triplets = []
        self.val_triplets = []

        for t in self.triplets:
            pos_files = list(t['positives'])
            neg_files = list(t['negatives'])

            random.shuffle(pos_files)
            random.shuffle(neg_files)

            val_p = pos_files[:self.val_pos]
            train_p = pos_files[self.val_pos:]

            val_n = neg_files[:self.val_neg]
            train_n = neg_files[self.val_neg:]

            self.train_triplets.append({
                'anchor': t['anchor'],
                'positives': train_p,
                'negatives': train_n,
                'protein_id': t['protein_id']
            })
            self.val_triplets.append({
                'anchor': t['anchor'],
                'positives': val_p,
                'negatives': val_n,
                'protein_id': t['protein_id']
            })

        total_train_p = sum(len(tr['positives']) for tr in self.train_triplets)
        total_train_n = sum(len(tr['negatives']) for tr in self.train_triplets)
        total_val_p = sum(len(vl['positives']) for vl in self.val_triplets)
        total_val_n = sum(len(vl['negatives']) for vl in self.val_triplets)

        print(f"Split (per-family hold-out, seed={self.seed}):")
        print(f"  Families: {len(self.triplets)}")
        print(f"  Train variants: {total_train_p} positives, {total_train_n} negatives")
        print(f"  Val variants  : {total_val_p} positives, {total_val_n} negatives")

    def save_split(self, path=None):
        """Save the current split to a JSON file for reproducibility."""
        if path is None:
            path = os.path.join(self.root_dir, self.SPLIT_FILENAME)

        split_data = {}
        for train_t, val_t in zip(self.train_triplets, self.val_triplets):
            pid = train_t['protein_id']
            split_data[pid] = {
                'val_positives': [os.path.basename(p) for p in val_t['positives']],
                'val_negatives': [os.path.basename(n) for n in val_t['negatives']],
            }

        with open(path, 'w') as f:
            json.dump({'seed': self.seed, 'val_pos': self.val_pos,
                       'val_neg': self.val_neg, 'families': split_data}, f, indent=2)

        print(f"Split saved to {path}")

    def _load_split(self, path):
        """Load a previously saved split from JSON."""
        with open(path, 'r') as f:
            saved = json.load(f)

        families_map = saved['families']
        self.train_triplets = []
        self.val_triplets = []

        loaded_count = 0
        for t in self.triplets:
            pid = t['protein_id']
            if pid not in families_map:
                self.train_triplets.append(t)
                self.val_triplets.append({
                    'anchor': t['anchor'], 'positives': [], 'negatives': [],
                    'protein_id': pid
                })
                continue

            saved_family = families_map[pid]
            val_pos_basenames = set(saved_family['val_positives'])
            val_neg_basenames = set(saved_family['val_negatives'])

            train_p = [p for p in t['positives'] if os.path.basename(p) not in val_pos_basenames]
            val_p = [p for p in t['positives'] if os.path.basename(p) in val_pos_basenames]
            train_n = [n for n in t['negatives'] if os.path.basename(n) not in val_neg_basenames]
            val_n = [n for n in t['negatives'] if os.path.basename(n) in val_neg_basenames]

            self.train_triplets.append({
                'anchor': t['anchor'], 'positives': train_p,
                'negatives': train_n, 'protein_id': pid
            })
            self.val_triplets.append({
                'anchor': t['anchor'], 'positives': val_p,
                'negatives': val_n, 'protein_id': pid
            })
            loaded_count += 1

        total_val_p = sum(len(vl['positives']) for vl in self.val_triplets)
        total_val_n = sum(len(vl['negatives']) for vl in self.val_triplets)
        print(f"Loaded split from {path} ({loaded_count} families matched)")
        print(f"  Val variants: {total_val_p} positives, {total_val_n} negatives")

    def get_split(self, split='train'):
        """Get triplets for a specific split."""
        if split == 'train':
            return self.train_triplets
        elif split == 'val':
            return self.val_triplets
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'")
