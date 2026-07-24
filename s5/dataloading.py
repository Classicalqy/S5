import torch
from pathlib import Path
import numpy as np
import os
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, Dict, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


# Example interface for making a loader.
def custom_loader(cache_dir: str,
				  bsz: int = 50,
				  seed: int = 42) -> ReturnType:
	...


def make_data_loader(dset,
					 dobj,
					 seed: int,
					 batch_size: int=128,
					 shuffle: bool=True,
					 drop_last: bool=True,
					 collate_fn: callable=None):
	"""

	:param dset: 			(PT dset):		PyTorch dataset object.
	:param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
	:param seed: 			(int):			Int for seeding shuffle.
	:param batch_size: 		(int):			Batch size for batches.
	:param shuffle:         (bool):			Shuffle the data loader?
	:param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
	:return:
	"""

	# Create a generator for seeding random number draws.
	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	if dobj is not None:
		assert collate_fn is None
		collate_fn = dobj._collate_fn

	# Generate the dataloaders.
	return torch.utils.data.DataLoader(dataset=dset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle,
									   drop_last=drop_last, generator=rng)


def create_lra_imdb_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										   bsz: int = 50,
										   seed: int = 42) -> ReturnType:
	"""

	:param cache_dir:		(str):		Not currently used.
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	:return:
	"""
	print("[*] Generating LRA-text (IMDB) Classification Dataset")
	from s5.dataloaders.lra import IMDB
	name = 'imdb'

	dataset_obj = IMDB('imdb', )
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trainloader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	testloader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	valloader = None

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = 135  # We should probably stop this from being hard-coded.
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trainloader, valloader, testloader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_listops_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											  bsz: int = 50,
											  seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-listops Classification Dataset")
	from s5.dataloaders.lra import ListOps
	name = 'listops'
	dir_name = './raw_datasets/lra_release/lra_release/listops-1000'

	dataset_obj = ListOps(name, data_dir=dir_name)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = 20
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_path32_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											 bsz: int = 50,
											 seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-Pathfinder32 Classification Dataset")
	from s5.dataloaders.lra import PathFinder
	name = 'pathfinder'
	resolution = 32
	dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_pathx_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											bsz: int = 50,
											seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-PathX Classification Dataset")
	from s5.dataloaders.lra import PathFinder
	name = 'pathfinder'
	resolution = 128
	dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_image_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											seed: int = 42,
											bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating LRA-listops Classification Dataset")
	from s5.dataloaders.basic import CIFAR10
	name = 'cifar'

	kwargs = {
		'grayscale': True,  # LRA uses a grayscale CIFAR image.
	}

	dataset_obj = CIFAR10(name, data_dir=cache_dir, **kwargs)  # TODO - double check what the dir here does.
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 32 * 32
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_aan_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										  bsz: int = 50,
										  seed: int = 42, ) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-AAN Classification Dataset")
	from s5.dataloaders.lra import AAN
	name = 'aan'

	dir_name = './raw_datasets/lra_release/lra_release/tsv_data'

	kwargs = {
		'n_workers': 1,  # Multiple workers seems to break AAN.
	}

	dataset_obj = AAN(name, data_dir=dir_name, **kwargs)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = len(dataset_obj.vocab)
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_speechcommands35_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
												   bsz: int = 50,
												   seed: int = 42) -> ReturnType:
	"""
	AG inexplicably moved away from using a cache dir...  Grumble.
	The `cache_dir` will effectively be ./raw_datasets/speech_commands/0.0.2 .

	See abstract template.
	"""
	print("[*] Generating SpeechCommands35 Classification Dataset")
	from s5.dataloaders.basic import SpeechCommands
	name = 'sc'

	dir_name = f'./raw_datasets/speech_commands/0.0.2/'
	os.makedirs(dir_name, exist_ok=True)

	kwargs = {
		'all_classes': True,
		'sr': 1  # Set the subsampling rate.
	}
	dataset_obj = SpeechCommands(name, data_dir=dir_name, **kwargs)
	dataset_obj.setup()
	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = 1
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	# Also make the half resolution dataloader.
	kwargs['sr'] = 2
	dataset_obj = SpeechCommands(name, data_dir=dir_name, **kwargs)
	dataset_obj.setup()
	val_loader_2 = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader_2 = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	aux_loaders = {
		'valloader2': val_loader_2,
		'testloader2': tst_loader_2,
	}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_speechcommands10_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											   bsz: int = 50,
											   seed: int = 42) -> ReturnType:
	"""Create the official Speech Commands v0.02 ten-command classification split."""
	print("[*] Generating SpeechCommands10 Classification Dataset (official split)")
	from s5.dataloaders.basic import SpeechCommands

	dir_name = './raw_datasets/speech_commands/0.0.2/'
	os.makedirs(dir_name, exist_ok=True)
	kwargs = {
		'all_classes': False,
		'split_policy': 'official',
		'sr': 1,
	}
	dataset_obj = SpeechCommands('sc', data_dir=dir_name, **kwargs)
	dataset_obj.setup()
	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	kwargs['sr'] = 2
	dataset_obj = SpeechCommands('sc', data_dir=dir_name, **kwargs)
	dataset_obj.setup()
	aux_loaders = {
		'valloader2': make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False),
		'testloader2': make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False),
	}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_cifar_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										seed: int = 42,
										bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating CIFAR (color) Classification Dataset")
	from s5.dataloaders.basic import CIFAR10
	name = 'cifar'

	kwargs = {
		'grayscale': False,  # LRA uses a grayscale CIFAR image.
	}

	dataset_obj = CIFAR10(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 32 * 32
	IN_DIM = 3
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_mnist_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
																				seed: int = 42,
																				bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating MNIST Classification Dataset")
	from s5.dataloaders.basic import MNIST
	name = 'mnist'

	kwargs = {
		'permute': False
	}

	dataset_obj = MNIST(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 28 * 28
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}
	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_pmnist_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
																				seed: int = 42,
																				bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating permuted-MNIST Classification Dataset")
	from s5.dataloaders.basic import MNIST
	name = 'mnist'

	kwargs = {
		'permute': True
	}

	dataset_obj = MNIST(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 28 * 28
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}
	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_synthetic_frequency_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
													  seed: int = 42,
													  bsz: int = 128,
													  seq_len: int = 256,
													  noise_std: float = 0.25,
													  low_freq_range=(4.0, 6.0),
													  high_freq_range=(7.0, 9.0),
													  amplitude_range=(0.5, 1.5),
													  bias_range=(-0.3, 0.3),
													  trend_range=(-0.3, 0.3),
													  distractor_count: int = 1,
													  distractor_freq_range=(1.0, 12.0),
													  distractor_amp_range=(0.0, 0.4),
													  num_train: int = 1000,
													  num_val: int = 200,
													  num_test: int = 200) -> ReturnType:
	"""Binary low-vs-high sinusoid classification for frequency selectivity.

	The default task is intentionally moderately noisy: nearby frequency bands,
	random amplitude, DC offset, slow drift, and an unrelated sinusoidal
	distractor.  This makes the task less solvable by simple magnitude or mean
	statistics and puts more pressure on frequency-selective dynamics.
	"""
	print("[*] Generating Synthetic Frequency Classification Dataset")

	def make_split(num_samples, split_seed):
		gen = torch.Generator()
		gen.manual_seed(split_seed)
		labels = torch.randint(0, 2, (num_samples,), generator=gen)
		low = torch.tensor(low_freq_range, dtype=torch.float32)
		high = torch.tensor(high_freq_range, dtype=torch.float32)
		amp_range = torch.tensor(amplitude_range, dtype=torch.float32)
		bias_rng = torch.tensor(bias_range, dtype=torch.float32)
		trend_rng = torch.tensor(trend_range, dtype=torch.float32)
		distractor_freq_rng = torch.tensor(distractor_freq_range, dtype=torch.float32)
		distractor_amp_rng = torch.tensor(distractor_amp_range, dtype=torch.float32)
		rand_freq = torch.rand(num_samples, generator=gen)
		low_freq = low[0] + rand_freq * (low[1] - low[0])
		high_freq = high[0] + rand_freq * (high[1] - high[0])
		freq = torch.where(labels == 0, low_freq, high_freq)
		phase = 2 * torch.pi * torch.rand(num_samples, generator=gen)
		t = torch.linspace(0.0, 1.0, seq_len)
		amplitude = amp_range[0] + torch.rand(num_samples, generator=gen) * (amp_range[1] - amp_range[0])
		bias = bias_rng[0] + torch.rand(num_samples, generator=gen) * (bias_rng[1] - bias_rng[0])
		trend = trend_rng[0] + torch.rand(num_samples, generator=gen) * (trend_rng[1] - trend_rng[0])
		signal = amplitude[:, None] * torch.sin(2 * torch.pi * freq[:, None] * t[None, :] + phase[:, None])

		for _ in range(distractor_count):
			d_freq = distractor_freq_rng[0] + torch.rand(num_samples, generator=gen) * (
				distractor_freq_rng[1] - distractor_freq_rng[0]
			)
			d_amp = distractor_amp_rng[0] + torch.rand(num_samples, generator=gen) * (
				distractor_amp_rng[1] - distractor_amp_rng[0]
			)
			d_phase = 2 * torch.pi * torch.rand(num_samples, generator=gen)
			signal = signal + d_amp[:, None] * torch.sin(
				2 * torch.pi * d_freq[:, None] * t[None, :] + d_phase[:, None]
			)

		signal = signal + bias[:, None] + trend[:, None] * (t[None, :] - 0.5)
		noise = noise_std * torch.randn(num_samples, seq_len, generator=gen)
		x = (signal + noise).unsqueeze(-1).float()
		y = labels.long()
		return torch.utils.data.TensorDataset(x, y)

	trainset = make_split(num_train, seed)
	valset = make_split(num_val, seed + 1)
	testset = make_split(num_test, seed + 2)

	trainloader = make_data_loader(trainset, None, seed=seed, batch_size=bsz)
	valloader = make_data_loader(valset, None, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	testloader = make_data_loader(testset, None, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	aux_loaders = {}
	return trainloader, valloader, testloader, aux_loaders, 2, seq_len, 1, num_train


def _resolve_ucr_split_file(data_dir: Union[str, Path],
							dataset_name: str,
							split: str) -> Path:
	root = Path(data_dir)
	split = split.upper()
	candidates = [
		root / dataset_name / f"{dataset_name}_{split}.tsv",
		root / dataset_name / f"{dataset_name}_{split}.TSV",
		root / f"{dataset_name}_{split}.tsv",
		root / f"{dataset_name}_{split}.TSV",
	]
	for path in candidates:
		if path.exists():
			return path

	for pattern in (f"{dataset_name}_{split}.tsv", f"{dataset_name}_{split}.TSV"):
		matches = sorted(root.rglob(pattern)) if root.exists() else []
		if matches:
			return matches[0]

	raise FileNotFoundError(
		f"Could not find UCR {dataset_name} {split} split under {root}. "
		f"Expected e.g. {root / dataset_name / f'{dataset_name}_{split}.tsv'}"
	)


def _load_ucr_tsv(path: Path,
				  label_to_index: Optional[Dict[float, int]] = None):
	data = np.loadtxt(path, delimiter="\t", dtype=np.float32)
	if data.ndim == 1:
		data = data[None, :]
	if data.shape[1] < 2:
		raise ValueError(f"UCR TSV file {path} must contain a label column and at least one feature column.")

	raw_labels = data[:, 0]
	if label_to_index is None:
		labels = sorted(float(label) for label in np.unique(raw_labels))
		label_to_index = {label: idx for idx, label in enumerate(labels)}

	unknown_labels = sorted(set(float(label) for label in np.unique(raw_labels)) - set(label_to_index))
	if unknown_labels:
		raise ValueError(f"UCR TSV file {path} contains labels not present in TRAIN split: {unknown_labels}")

	mapped_labels = np.array([label_to_index[float(label)] for label in raw_labels], dtype=np.int64)
	x = torch.from_numpy(data[:, 1:]).float().unsqueeze(-1)
	y = torch.from_numpy(mapped_labels).long()
	return torch.utils.data.TensorDataset(x, y), label_to_index


def create_ucr_classification_dataset(dataset_name: str,
									  cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
									  seed: int = 42,
									  bsz: int = 128,
									  val_split: float = 0.1,
									  split_mode: str = "standard") -> ReturnType:
	print(f"[*] Generating UCR-{dataset_name} Classification Dataset ({split_mode} split)")

	train_path = _resolve_ucr_split_file(cache_dir, dataset_name, "TRAIN")
	test_path = _resolve_ucr_split_file(cache_dir, dataset_name, "TEST")
	full_trainset, label_to_index = _load_ucr_tsv(train_path)
	full_testset, _ = _load_ucr_tsv(test_path, label_to_index)

	if not 0.0 < val_split < 1.0:
		raise ValueError(f"val_split must be between 0 and 1, got {val_split}")

	if split_mode == "standard":
		val_size = max(1, int(len(full_trainset) * val_split))
		train_size = len(full_trainset) - val_size
		if train_size < 1:
			raise ValueError(f"UCR {dataset_name} TRAIN split is too small for val_split={val_split}")

		trainset, valset = torch.utils.data.random_split(
			full_trainset,
			(train_size, val_size),
			generator=torch.Generator().manual_seed(seed),
		)
		testset = full_testset
	elif split_mode == "combined":
		full_dataset = torch.utils.data.ConcatDataset([full_trainset, full_testset])
		total_size = len(full_dataset)
		train_size = int(0.8 * total_size)
		val_size = int(0.1 * total_size)
		test_size = total_size - train_size - val_size
		if min(train_size, val_size, test_size) < 1:
			raise ValueError(f"UCR {dataset_name} combined split is too small for 80/10/10.")

		trainset, valset, testset = torch.utils.data.random_split(
			full_dataset,
			(train_size, val_size, test_size),
			generator=torch.Generator().manual_seed(seed),
		)
	else:
		raise ValueError(f"Unknown UCR split_mode {split_mode}. Expected 'standard' or 'combined'.")

	trainloader = make_data_loader(trainset, None, seed=seed, batch_size=bsz)
	valloader = make_data_loader(valset, None, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	testloader = make_data_loader(testset, None, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	seq_len = full_trainset.tensors[0].shape[1]
	n_classes = len(label_to_index)
	aux_loaders = {}
	return trainloader, valloader, testloader, aux_loaders, n_classes, seq_len, 1, train_size


def create_ucr_ecg5000_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											  seed: int = 42,
											  bsz: int = 128,
											  split_mode: str = "standard") -> ReturnType:
	return create_ucr_classification_dataset("ECG5000", cache_dir=cache_dir, seed=seed, bsz=bsz, split_mode=split_mode)


def create_ucr_forda_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											seed: int = 42,
											bsz: int = 128,
											split_mode: str = "standard") -> ReturnType:
	return create_ucr_classification_dataset("FordA", cache_dir=cache_dir, seed=seed, bsz=bsz, split_mode=split_mode)


def create_ucr_wafer_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											seed: int = 42,
											bsz: int = 128,
											split_mode: str = "standard") -> ReturnType:
	return create_ucr_classification_dataset("Wafer", cache_dir=cache_dir, seed=seed, bsz=bsz, split_mode=split_mode)


Datasets = {
	# Other loaders.
	"mnist-classification": create_mnist_classification_dataset,
	"pmnist-classification": create_pmnist_classification_dataset,
	"cifar-classification": create_cifar_classification_dataset,
	"synthetic_frequency-classification": create_synthetic_frequency_classification_dataset,
	"ucr-ecg5000-classification": create_ucr_ecg5000_classification_dataset,
	"ucr-forda-classification": create_ucr_forda_classification_dataset,
	"ucr-wafer-classification": create_ucr_wafer_classification_dataset,

	# LRA.
	"imdb-classification": create_lra_imdb_classification_dataset,
	"listops-classification": create_lra_listops_classification_dataset,
	"aan-classification": create_lra_aan_classification_dataset,
	"lra-cifar-classification": create_lra_image_classification_dataset,
	"pathfinder-classification": create_lra_path32_classification_dataset,
	"pathx-classification": create_lra_pathx_classification_dataset,

	# Speech.
	"speech10-classification": create_speechcommands10_classification_dataset,
	"speech35-classification": create_speechcommands35_classification_dataset,
}
