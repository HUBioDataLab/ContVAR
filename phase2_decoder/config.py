class DecoderConfig:

    def __init__(self, aspect: str = "F"):
        # aspect: "F"=MF | "P"=BP | "C"=CC  (GO namespace)
        self.aspect = aspect
        asp = aspect.lower()

        # Encoder
        # 1280 when using ESM-2 650M directly; 256 for ContVAR GNN encoder output
        self.encoder_output_dim   = 1280

        # Decoder architecture
        self.hidden_dims          = [512, 1024, 512]
        self.dropout              = 0.3

        # GO classification
        self.min_go_freq          = 10
        self.null_function_weight = 3.0
        self.pos_weight_clamp     = 20.0

        # Training
        self.lr                   = 1e-3
        self.weight_decay         = 1e-4
        self.batch_size           = 256
        self.epochs               = 100
        self.early_stop_patience  = 10
        self.seed                 = 42
        self.eval_threshold       = 0.95

        # File paths — aspect-specific
        self.goa_tsv              = "goa_2025-12-04_swissprot_noiea.tsv"
        self.embeddings_h5        = "esm2_t33_650M_UR50D_protein_embedding.h5"
        self.go_vocab_json        = f"go_vocab_{asp}.json"
        self.decoder_checkpoint   = f"decoder_best_{asp}.pt"
        self.uniref_tsv           = "protein_uniref50.tsv"
        self.split_json           = "phase0_go_split.json"

        # Weights & Biases
        self.wandb_project        = "ContVAR-Project"
        self.wandb_entity         = "canerayvaz-hacettepe-university"
        self.wandb_run_name       = f"decoder-{asp}"
        self.wandb_api_key        = None
