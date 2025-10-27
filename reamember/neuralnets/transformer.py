from transformers import AutoModel, AutoTokenizer

class Transformer:
    def __init__(self, model_name="distilbert/distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.latent_dim = self.model.config.hidden_size

    def encode(self, texts, device=None):
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Get the embeddings from the last hidden state
        return outputs.last_hidden_state
    
    def decode(self, embeddings):
        inputs = self.tokenizer.batch_decode(embeddings, skip_special_tokens=True)
        return inputs

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.model = AutoModel.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        return model