import transformers
import torch
import torch.nn.functional as F

print(f"Transformers version {transformers.__version__}")
print(f" PyTorch version {torch.__version__}")


class Encoder:
    def __init__(self):
        self.model = transformers.CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, query: str):
        tokenized_query: transformers.tokenization_utils_base.BatchEncoding = (
            self.tokenizer([query], padding=True, return_tensors="pt")
        )
        print(tokenized_query)
        embeddings = F.normalize(self.model.get_text_features(**tokenized_query))
        return embeddings

    def process_image(self, image):
        processed_image = self.processor(
            images=image, return_tensors="pt", padding=True
        )
        embeddings_image = F.normalize(self.model.get_image_features(**processed_image))
        return embeddings_image
