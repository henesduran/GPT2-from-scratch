import torch, tiktoken
from transformers import GPT2LMHeadModel
from src.model import GPT

def test_logits_match():
    torch.manual_seed(0)
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor([enc.encode("The capital of France is")], dtype=torch.long)

    my_model = GPT.from_pretrained("gpt2")
    my_model.eval()

    hf_model = GPT2LMHeadModel.from_pretrained("gpt2") 
    hf_model.eval()

    with torch.no_grad():
        my_logits, _ = my_model(tokens) #not interested in loss
        hf_logits = hf_model(tokens).logits

    max_diff = (my_logits - hf_logits).abs().max().item()
    print(f"max abs logit diff: {max_diff:.2e}")
    assert max_diff < 1e-3

if __name__ == "__main__":
    test_logits_match()