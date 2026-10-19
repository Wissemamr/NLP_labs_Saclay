"""Offline integration check of notebook training, figures, and artifact persistence."""
import csv
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ['MPLBACKEND'] = 'Agg'
import torch
from torch.utils.data import DataLoader


class TinyTokenizer:
    eos_token_id = 31

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        sequences = [[1, 2, 3, 4][:2 + len(text) % 3] for text in texts]
        length = max(map(len, sequences))
        ids = torch.full((len(texts), length), 31, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence)
            mask[index, :len(sequence)] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)

    def save_pretrained(self, path):
        path.mkdir(parents=True)
        (path / 'test_tokenizer.json').write_text('{}')

    def decode(self, tokens, **kwargs):
        return ' '.join(map(str, tokens))


def main():
    root = Path(__file__).parent
    notebook = json.loads((root / 'implement_transformer.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    for index in (1, 3, 6, 8, 10, 12, 25):
        exec(''.join(notebook['cells'][index]['source']), namespace)
    config = dict(vocabulary_size=32, embed_size=16, intermediate_size=32, num_heads=2, hidden_layers=2)
    training = [{'text': str(i)} for i in range(5)]
    validation = [{'text': 'a'}, {'text': 'bb'}, {'text': 'ccc'}]
    with tempfile.TemporaryDirectory() as directory:
        namespace.update(model_config=config, device=torch.device('cpu'), tokenizer=TinyTokenizer(),
                         model_hg_id='synthetic-offline-test', RUNS_DIR=Path(directory), MAX_LENGTH=8,
                         MAX_STEPS=4, NUM_EPOCHS=2, EVAL_EVERY=3,
                         training_dl=DataLoader(training, batch_size=2, shuffle=True,
                                                generator=torch.Generator().manual_seed(42)),
                         validation_dl=DataLoader(validation, batch_size=2))
        run_ids = []
        for _ in range(2):
            for index in (23, 26, 28):
                exec(''.join(notebook['cells'][index]['source']), namespace)
            run_dir = namespace['RUN_DIR']
            run_ids.append(run_dir.name)
            metadata = json.loads((run_dir / 'metadata.json').read_text())
            assert metadata['status'] == 'completed'
            assert metadata['steps_completed'] == 4
            assert metadata['samples_processed'] == 7
            assert metadata['unique_training_samples_seen'] == 5
            assert metadata['epochs_completed'] == 1.4
            with (run_dir / 'metrics.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            assert [int(r['step']) for r in rows if r['split'] == 'validation'] == [0, 3, 4]
            assert len([r for r in rows if r['split'] == 'train']) == 4
            for name in ('model.pt', 'metadata.json', 'training.log', 'sources.json',
                         'learning_curves.png', 'learning_curves.pdf', 'generations.jsonl'):
                assert (run_dir / name).stat().st_size > 0, name
            checkpoint = torch.load(run_dir / 'model.pt', weights_only=True)
            assert checkpoint['step'] == 4
            torch.testing.assert_close(namespace['model'].wte.weight,
                                       checkpoint['model_state_dict']['wte.weight'])
        assert run_ids[0] != run_ids[1]
        # Verify evaluations restore the model's training state.
        namespace['model'].train()
        from experiment import evaluate, train_run
        evaluate(namespace['model'], namespace['validation_dl'], namespace['tokenizer'], 8, 'cpu')
        assert namespace['model'].training
        # An invalid batch is logged as a failed run, retaining the baseline metrics.
        class EmptyTargets(TinyTokenizer):
            def __call__(self, texts, **kwargs):
                result = super().__call__(texts, **kwargs)
                if list(texts) == ['bad']:
                    result.attention_mask.zero_()
                return result
        try:
            train_run(namespace['model'], namespace['optimizer'],
                      DataLoader([{'text': 'bad'}], batch_size=1), namespace['validation_dl'],
                      EmptyTargets(), model_config=config, runs_dir=directory, max_length=8,
                      max_steps=1, num_epochs=1, eval_every=1, seed=42, tokenizer_id='test')
        except ValueError:
            pass
        else:
            raise AssertionError('Expected invalid batch to fail')
        states = [json.loads(p.read_text())['status'] for p in Path(directory).glob('*/metadata.json')]
        assert sorted(states) == ['completed', 'completed', 'failed']
    print('PASS: notebook artifacts, partial/multiple epochs, timestamp isolation, metrics, figures, reload, failure logging')


if __name__ == '__main__':
    main()
