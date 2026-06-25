# LTSpice Export For Hardware-Friendly S5 Layers

This package exports trained hardware-friendly S5 SSM layers to LTSpice netlists.
The layer exporter supports `real_decay`, `resonant_2x2`, and `energy_shaped_2x2`
`RealValuedSSM` modules. Dense encoder/decoder layers, normalization,
residual paths, dropout, GLU/GELU activations, and classifier heads are skipped
by the layer-only exporter.

## Usage

```bash
python -m spice.export_netlist \
  --params model.msgpack \
  --ssm-param real_decay \
  --sample-rate 16000 \
  --out out/model.cir
```

The command writes:

- `model.cir`: an LTSpice netlist with ideal op-amps, unity inverters, RC state
  blocks, and output add/sub stages.
- `model_components.json`: an audit manifest containing every layer, 2x2 block,
  continuous-time coefficient, resistor, capacitor, and node name used by the
  exporter.

## Parameter Mapping

For each SSM module, the exporter recursively looks for:

```text
B, C, raw_alpha, log_step
```

and, for the 2x2 modes, also:

```text
omega
```

and, for `energy_shaped_2x2`, also:

```text
raw_q
```

The continuous-time parameters are reconstructed as:

```text
Delta = exp(log_step[:, 0])
alpha = softplus(raw_alpha) + 1e-4

# real_decay
A_i = [-alpha_i]
B_tr,i = sample_rate * Delta_i * B_i

# resonant_2x2 / energy_shaped_2x2
q = softplus(raw_q) + 1e-4       # energy_shaped_2x2 only
q = 1                            # resonant_2x2

A_k = q_k [[-alpha_k, -omega_k],
           [ omega_k, -alpha_k]]

A_tr,k = sample_rate * Delta_k * A_k
B_tr,k = sample_rate * Delta_k * B.reshape(n_blocks, 2, H)[k]
C_tr = C
```

For every nonzero continuous-time gain `g` feeding an inverting active
low-pass/integrator summing node, the generated resistor is:

```text
R = 1 / (abs(g) * C_state)
```

The default state capacitor is `1uF`, configurable with
`--state-capacitance`.

## Circuit Shape

Each `real_decay` state is emitted as a one-state active low-pass/integrator
subcircuit with a decay resistor and input-weight resistors. Each 2x2 block is
emitted as a named subcircuit with:

- two ideal-op-amp active low-pass/integrator states,
- one unity inverter for the positive cross-coupling branch,
- RC feedback for decay terms,
- cross-coupling resistors for `omega`,
- input-weight resistors for `B_tr`.

Each layer output is emitted as a sign-aware add/sub stage derived from the
paper's resistor rules. When needed, a grounded dummy negative branch is added
so all generated resistor values remain positive.

## Restricted Full MNIST Export

`spice.export_full_model` adds a minimal full-model netlist for the current
restricted MNIST checkpoint:

```text
Dense encoder -> SSM0 -> ReLU -> SSM1 -> ReLU -> Dense decoder logits
```

It assumes exactly two hardware-friendly SSM layers, `activation_fn=relu`,
`mode=last`, no residual path, no batch/layer norm, no dropout, and decoder
logits only. These assumptions are recorded in the generated component
manifest, but they are not all recoverable from a Flax params-only checkpoint.
Use a checkpoint trained with the matching config.

The generated full circuit is a continuous analog cascade:

```text
circuit_semantics = continuous_cascade_without_inter_layer_sample_hold
```

That means `SSM0 -> ReLU -> SSM1` is wired directly in continuous time. This is
not exactly the same semantics as the digital stacked SSM recurrence, where each
layer consumes a sampled sequence. For exact digital stacked semantics in
hardware, an inter-layer sample-and-hold stage is needed.

## Validation Modes

For the full export-and-validation flow, use:

```bash
python -m spice.workflow \
  --params checkpoints/mnist_resonant_2x2_seed0_params.msgpack \
  --ssm-param resonant_2x2 \
  --sample-rate 16000 \
  --out-dir out/spice_workflow \
  --full-samples 5 \
  --accuracy-samples 100 \
  --delete-raw-after-read \
  --delete-log-after-read
```

The workflow writes:

- `netlists/ssm_layers.cir` and `netlists/full_model.cir`,
- `layer_sanity/summary.json` plus per-layer Python reference CSV, LTSpice
  deck, rRMSE metrics, and state overlay PNGs,
- `full_alignment/summary.json`, per-sample digital/continuous references,
  per-layer state/output rRMSE tables, and per-sample plots for combined
  first-block states, combined per-layer worst states, and final logits when
  LTSpice `.raw` files are present,
- `accuracy/summary.json` and `accuracy/per_sample.csv` for logit-only test-set
  accuracy plus digital/LTSpice margin analysis.

Use `--no-run-ltspice` to only generate netlists, decks, references, and pending
summaries. Re-running the same command with the same `--out-dir` resumes
completed accuracy samples and reuses existing `.raw` files.

`spice.validate_digital_alignment` compares three references:

- digital ZOH SSM recurrence used by training,
- Python continuous cascade,
- LTSpice transient sampled at the digital sample times, when `.raw` files are
  present.

Use it to diagnose whether an error is a circuit/export issue or a semantics
difference between continuous cascades and sampled stacked recurrences.

`spice.validate_ltspice_accuracy` is a logit-only MNIST accuracy runner for
larger subsets. It saves only `V(LOGIT0)` through `V(LOGIT9)`, writes
`per_sample.csv`, can resume completed samples, and can delete `.raw`/`.log`
files after reading final logits.

Its logit-difference fields are named `ltspice_vs_digital_*` intentionally:
they compare the LTSpice continuous cascade final logits against the digital
stacked recurrence final logits. Those differences therefore include both
analog transient error and the expected model-semantics difference.
