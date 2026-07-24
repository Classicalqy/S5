from pathlib import Path

import torch

from s5 import dataloading
from s5.dataloaders import basic, sc


def _loader_instance(root, split_policy="official", all_classes=False):
    dataset = object.__new__(sc._SpeechCommands)
    dataset.root = Path(root)
    dataset.dropped_rate = 0.0
    dataset.split_policy = split_policy
    dataset.all_classes = all_classes
    dataset.gen = False
    dataset.discrete_input = False
    return dataset


def test_sc10_official_cache_is_distinct_from_legacy_and_sc35(tmp_path):
    base = tmp_path / "SpeechCommands" / "processed_data"
    official = _loader_instance(tmp_path, "official")._data_location(base, mfcc=False)
    legacy = _loader_instance(tmp_path, "stratified")._data_location(base, mfcc=False)
    all_classes = _loader_instance(tmp_path, "official", all_classes=True)._data_location(base, mfcc=False)

    assert official.name == "raw_sc10_official"
    assert legacy.name == "raw"
    assert all_classes.name == "raw_all_classes"


def test_sc10_official_processing_uses_lists_labels_and_zero_padding(monkeypatch, tmp_path):
    data_root = tmp_path / "SpeechCommands"
    validation_paths, testing_paths = [], []
    for label, word in enumerate(sc._SpeechCommands.SUBSET_CLASSES):
        word_dir = data_root / word
        word_dir.mkdir(parents=True)
        for split in ("train", "val", "test"):
            filename = f"{split}.wav"
            (word_dir / filename).touch()
            relative_path = f"{word}/{filename}"
            if split == "val":
                validation_paths.append(relative_path)
            elif split == "test":
                testing_paths.append(relative_path)

    (data_root / "validation_list.txt").write_text("\n".join(validation_paths) + "\n")
    (data_root / "testing_list.txt").write_text("\n".join(testing_paths) + "\n")

    def fake_load(path, channels_first=False):
        assert channels_first is False
        sample_len = 4 if path.name == "val.wav" else 6
        return torch.ones((sample_len, 1)), 16000

    monkeypatch.setattr(sc.torchaudio, "load", fake_load)
    dataset = _loader_instance(tmp_path, "official")
    train_X, val_X, test_X, train_y, val_y, test_y = dataset._process_official_subset(mfcc=False)

    # Cached raw tensors are channel-first; ``load_data`` transposes them to
    # the [batch, time, channel] shape consumed by the training loop.
    assert train_X.shape == val_X.shape == test_X.shape == (10, 1, 16000)
    assert train_y.tolist() == val_y.tolist() == test_y.tolist() == list(range(10))
    padded = sc._SpeechCommands._pad_or_trim_audio(torch.ones((4, 1)))
    assert torch.equal(padded[:4], torch.ones((4, 1)))
    assert torch.count_nonzero(padded[4:]) == 0


def test_speech10_factory_uses_official_raw_configuration(monkeypatch, tmp_path):
    calls = []

    class FakeSpeechCommands:
        _collate_fn = staticmethod(torch.utils.data.default_collate)

        def __init__(self, name, data_dir, **kwargs):
            calls.append(kwargs)
            inputs = torch.zeros((4, 16000, 1))
            targets = torch.tensor([0, 1, 2, 3])
            self.dataset_train = torch.utils.data.TensorDataset(inputs, targets)
            self.dataset_val = torch.utils.data.TensorDataset(inputs, targets)
            self.dataset_test = torch.utils.data.TensorDataset(inputs, targets)
            self.d_output = 10
            self.d_input = 1

        def setup(self):
            return None

    monkeypatch.setattr(basic, "SpeechCommands", FakeSpeechCommands)
    monkeypatch.setattr(dataloading.os, "makedirs", lambda *args, **kwargs: None)
    result = dataloading.create_speechcommands10_classification_dataset(bsz=2, seed=7)
    trainloader, _valloader, _testloader, aux_loaders, classes, sequence_length, input_dim, train_size = result

    assert calls == [
        {"all_classes": False, "split_policy": "official", "sr": 1},
        {"all_classes": False, "split_policy": "official", "sr": 2},
    ]
    assert classes == 10
    assert sequence_length == 16000
    assert input_dim == 1
    assert train_size == 4
    assert set(aux_loaders) == {"valloader2", "testloader2"}
    batch_inputs, batch_targets = next(iter(trainloader))
    assert batch_inputs.shape == (2, 16000, 1)
    assert batch_targets.shape == (2,)
    assert "speech10-classification" in dataloading.Datasets
