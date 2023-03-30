import transformers
import torch
import torch.nn.functional as F


class Encoder:
    def __init__(self):
        self.model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/msmarco-distilbert-base-v2"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/msmarco-distilbert-base-v2"
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = (
            model_output.last_hidden_state
        )  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts):
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
