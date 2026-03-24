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
        device="cpu",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.encoder_name = encoder
        self.decoder_name = decoder
        self.tokenizer_name = tokenizer
        self.device = torch.device(device)

        self.encode_model = TextToEmbeddingModelPipeline(
            encoder=self.encoder_name,
            tokenizer=self.tokenizer_name,
            device=self.device,
        )
        self.decode_model = EmbeddingToTextModelPipeline(
            decoder=self.decoder_name,
            tokenizer=self.tokenizer_name,
            device=self.device,
        )

    def encode(
        self,
        texts,
        max_seq_len=128,
        source_lang="eng_Latn",
        batch_size=32,
        device=None,
    ):
        if device is None:
            device = self.device

        if isinstance(texts, str):
            texts = [texts]

        if torch.is_tensor(texts):
            texts = [x for x in texts.tolist()]

        if not isinstance(texts, (list, tuple)):
            raise ValueError("`texts` debe ser una lista o tupla de strings")

        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings = self.encode_model.predict(
                batch,
                source_lang=source_lang,
                max_seq_len=max_seq_len,
            )

            if torch.is_tensor(embeddings):
                outputs.append(embeddings.detach().cpu())
            elif isinstance(embeddings, list):
                outputs.append(torch.tensor(embeddings, dtype=torch.float32))
            else:
                outputs.append(torch.from_numpy(embeddings).float())

        if len(outputs) == 0:
            return torch.empty((0, self.latent_dim), dtype=torch.float32)

        return torch.cat(outputs, dim=0)

    def decode(
        self,
        embeddings,
        target_lang="eng_Latn",
        max_seq_len=256,
        batch_size=32,
    ):
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()

        if isinstance(embeddings, (list, tuple)) and len(embeddings) > 0 and torch.is_tensor(embeddings[0]):
            embeddings = torch.stack(embeddings).cpu().numpy()

        if embeddings is None or len(embeddings) == 0:
            return []

        outputs = []
        for start in range(0, len(embeddings), batch_size):
            batch = embeddings[start : start + batch_size]
            texts = self.decode_model.predict(
                batch,
                target_lang=target_lang,
                max_seq_len=max_seq_len,
            )
            if isinstance(texts, str):
                outputs.append(texts)
            else:
                outputs.extend(texts)

        return outputs

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
            "device": str(self.device),
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
            device=device,
        )
