# LTSpice Export For Hardware-Friendly S5 Layers

This package exports trained hardware-friendly S5 SSM layers to LTSpice netlists.
The first version supports only `resonant_2x2` and `energy_shaped_2x2`
`RealValuedSSM` modules. Dense encoder/decoder layers, normalization,
residual paths, dropout, GLU/GELU activations, and classifier heads are skipped.

## Usage

```bash
python -m spice.export_netlist \
  --params model.msgpack \
  --ssm-param resonant_2x2 \
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
B, C, raw_alpha, omega, log_step
```

and, for `energy_shaped_2x2`, also:

```text
raw_q
```

The continuous-time parameters are reconstructed as:

```text
alpha = softplus(raw_alpha) + 1e-4
q = softplus(raw_q) + 1e-4       # energy_shaped_2x2 only
q = 1                            # resonant_2x2
Delta = exp(log_step[:, 0])

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

Each 2x2 block is emitted as a named subcircuit with:

- two ideal-op-amp active low-pass/integrator states,
- one unity inverter for the positive cross-coupling branch,
- RC feedback for decay terms,
- cross-coupling resistors for `omega`,
- input-weight resistors for `B_tr`.

Each layer output is emitted as a sign-aware add/sub stage derived from the
paper's resistor rules. When needed, a grounded dummy negative branch is added
so all generated resistor values remain positive.

