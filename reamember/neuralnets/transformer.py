from pathlib import Path

import torch
import pytorch_lightning as pl
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline, TextToEmbeddingModelPipeline


class SONAR(pl.LightningModule):
    def __init__(
        self,
        input_shape=None,
        latent_dim=1024,
        model_name="text_sonar_basic",
        encoder="text_sonar_basic_encoder",
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        max_seq_len=128,
        device="cpu",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.encoder_name = encoder
        self.decoder_name = decoder
        self.tokenizer_name = tokenizer
        self.max_seq_len = max_seq_len
        self.pipeline_device = torch.device(device)

        self.encode_model = TextToEmbeddingModelPipeline(
            encoder=self.encoder_name,
            tokenizer=self.tokenizer_name,
            device=self.pipeline_device,
        )
        self.decode_model = EmbeddingToTextModelPipeline(
            decoder=self.decoder_name,
            tokenizer=self.tokenizer_name,
            device=self.pipeline_device,
        )

    def encode(self, x, **kwargs):
        x = self.encode_model.predict(
            x,
            source_lang="eng_Latn",
            max_seq_len=self.max_seq_len,
        )
        return x

    def decode(self, z, **kwargs):
        z = self.decode_model.predict(
            z,
            target_lang="eng_Latn",
            max_seq_len=self.max_seq_len,
        )
        return z

    def forward(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    def save(self, path):
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        metadata = {
            "model_name": self.model_name,
            "encoder": self.encoder_name,
            "decoder": self.decoder_name,
            "tokenizer": self.tokenizer_name,
            "latent_dim": self.latent_dim,
            "input_shape": self.input_shape,
            "max_seq_len": self.max_seq_len,
            "device": str(self.pipeline_device),
        }

        torch.save(metadata, path)
        return path

    def load_state_dict(self, state_dict, strict=True):
        # No hay parámetros entrenables reusables en SONAR (todo viene del pipeline remoto).
        if isinstance(state_dict, dict) and "encoder" in state_dict:
            self.encoder_name = state_dict.get("encoder", self.encoder_name)
            self.decoder_name = state_dict.get("decoder", self.decoder_name)
            self.tokenizer_name = state_dict.get("tokenizer", self.tokenizer_name)
            self.latent_dim = state_dict.get("latent_dim", self.latent_dim)
        return {}

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {path}")

        metadata = torch.load(path, map_location=device)
        return cls(
            input_shape=metadata.get("input_shape", None),
            latent_dim=metadata.get("latent_dim", 1024),
            model_name=metadata.get("model_name", "text_sonar_basic"),
            encoder=metadata.get("encoder", "text_sonar_basic_encoder"),
            decoder=metadata.get("decoder", "text_sonar_basic_decoder"),
            tokenizer=metadata.get("tokenizer", "text_sonar_basic_encoder"),
            max_seq_len=metadata.get("max_seq_len", 128),
            device=device,
        )
