"""Offline checks of the actual notebook implementation: python smoke_check.py."""
import json
import tempfile
from pathlib import Path

import torch


def main():
    notebook = json.loads((Path(__file__).parent / 'implement_transformer.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    for index in (1, 3, 4, 6, 8, 10, 12, 25):
        exec(compile(''.join(notebook['cells'][index]['source']), f'cell_{index}', 'exec'), namespace)
    model_class = namespace['TransformerDecoder']
    model = model_class(32, 16, 32, 2, 2)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _, logits = model(tokens)
    assert logits.shape == (2, 4, 32)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1))
    before = model.wte.weight.detach().clone()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.wte.weight)
    model.eval()
    with torch.inference_mode():
        _, original = model(tokens)
        changed = tokens.clone()
        changed[:, -1] = 9
        _, altered = model(changed)
        torch.testing.assert_close(original[:, :-1], altered[:, :-1])
        # Changing a masked key cannot change a later, unmasked query.
        mask = torch.tensor([[1, 0, 1, 1], [1, 0, 1, 1]])
        _, masked = model(tokens, attention_mask=mask)
        changed = tokens.clone()
        changed[:, 1] = 10
        _, altered = model(changed, attention_mask=mask)
        torch.testing.assert_close(masked[:, 2:], altered[:, 2:])
        keys = torch.zeros(2, 2, 2, 16, 8)
        values = torch.zeros_like(keys)
        cached = []
        for position in range(tokens.shape[1]):
            _, output = model(tokens[:, position:position+1], k_cache=keys,
                              v_cache=values, start_cache=position)
            cached.append(output)
        torch.testing.assert_close(torch.cat(cached, dim=1), original, atol=1e-6, rtol=1e-5)
        greedy = namespace['nonefficient_greedy_decoding'](model, tokens, max_new_tokens=4)
        cached_greedy = namespace['greedy_decoding'](model, tokens, max_new_tokens=4)
        assert torch.equal(greedy, cached_greedy)
        sampled = namespace['sampling_decoding'](model, tokens, max_new_tokens=3)
        assert sampled.shape == (2, 7)
        # Exercise RoPE growth with a nonzero cached position.
        rope = namespace['RoPE'](8, precomp_len=2)
        output = rope(torch.randn(2, 2, 1, 8), start_pos_id=5)
        assert output.shape == (2, 2, 1, 8) and torch.isfinite(output).all()
        single_position = namespace['RoPE'](8, precomp_len=1)
        sample = torch.randn(2, 2, 1, 8)
        torch.testing.assert_close(single_position(sample), sample)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            torch.save(model.state_dict(), checkpoint)
            restored = model_class(32, 16, 32, 2, 2).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            torch.testing.assert_close(restored(tokens)[1], original)
    print('PASS: gradients, optimizer update, causality, padding, KV cache, decoding, RoPE offsets, checkpoint reload')


if __name__ == '__main__':
    main()
