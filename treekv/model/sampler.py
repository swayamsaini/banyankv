import torch
import torch.nn.functional as F


def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0.0:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    return F.softmax(logits.float() / temperature, dim=-1)


def sample_token(logits: torch.Tensor, temperature: float = 0.0) -> int:
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())
    probs = logits_to_probs(logits, temperature=temperature)
    return int(torch.multinomial(probs, num_samples=1).item())


def speculative_verify(
    draft_tokens: list[int],
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[list[int], int]:
    gamma = len(draft_tokens)
    accepted: list[int] = []

    for i in range(gamma):
        tok = draft_tokens[i]
        if temperature <= 0.0:
            target_tok = int(target_probs[i].argmax().item())
            if tok == target_tok:
                accepted.append(tok)
            else:
                accepted.append(target_tok)
                return accepted, i
        else:
            p_val = target_probs[i, tok].item()
            q_val = draft_probs[i, tok].item()
            accept_prob = min(1.0, p_val / (q_val + 1e-12))
            if torch.rand(1).item() < accept_prob:
                accepted.append(tok)
            else:
                residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                res_sum = residual.sum()
                if res_sum.item() > 1e-12:
                    residual = residual / res_sum
                    fixed_tok = int(torch.multinomial(residual, num_samples=1).item())
                else:
                    fixed_tok = int(torch.multinomial(target_probs[i], num_samples=1).item())
                accepted.append(fixed_tok)
                return accepted, i

    if temperature <= 0.0:
        bonus = int(target_probs[gamma].argmax().item())
    else:
        bonus = int(torch.multinomial(target_probs[gamma], num_samples=1).item())
    accepted.append(bonus)
    return accepted, gamma
