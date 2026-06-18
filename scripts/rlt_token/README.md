# RLT Stage 1: RL Token Training

This folder implements the first stage of the RLT-style pipeline for our UR3e SmolVLA model.

For the full UR3e RLT workflow and rollout/training commands, start from
`scripts/rlt_train/README.md`.

The goal of Stage 1 is not to control the robot yet. The goal is to teach a small network to compress what the frozen VLA knows into a compact vector:

```text
frozen SmolVLA internal embeddings -> RLT encoder -> z_rl
z_rl -> RLT decoder -> reconstruct frozen SmolVLA embeddings
```

After this stage, `z_rl` is the RL token representation we can feed into later RL actors/critics.

## Why Stage 1 Exists

SmolVLA is large. It sees images, language, state, and predicts an action chunk. If we use the full VLA directly inside online RL, it is expensive and awkward:

- the representation is high-dimensional;
- RL updates can destabilize the VLA if we train it directly;
- online control wants a compact state representation.

RLT solves this by freezing the VLA and learning a small token encoder. The encoder learns to summarize important VLA information into `z_rl`. A decoder is trained only as a learning signal: if `z_rl` can reconstruct the VLA embeddings, it probably retained useful information.

## What We Extract From SmolVLA

Our SmolVLA has two relevant internal streams:

- `VLM / prefix stream`: image + language + robot state context.
- `expert / suffix stream`: action chunk tokens produced from the demonstration action sequence.

In `train_rlt_stage1.py`, we:

1. Load the trained SmolVLA checkpoint.
2. Freeze every SmolVLA parameter.
3. Load the same LeRobot dataset used by that checkpoint.
4. Request an action chunk from LeRobot using:

```text
action timestamps = [0/fps, 1/fps, ..., (chunk_size-1)/fps]
```

5. Run SmolVLA internally:

```text
images + language + state -> prefix/context embeddings
action chunk -> suffix/expert embeddings
```

6. Pool each stream into one vector:

```text
vlm_embedding    shape: 960
expert_embedding shape: 720
```

7. Train the Stage 1 autoencoder:

```text
[vlm_embedding, expert_embedding, learnable_RL_query]
        -> small Transformer encoder
        -> z_rl, default shape 256
        -> decoder
        -> reconstruct vlm_embedding and expert_embedding
```

The reconstruction loss is:

```text
loss = MSE(vlm_recon, vlm_embedding)
     + expert_loss_weight * MSE(expert_recon, expert_embedding)
```

## Important Mental Model

`z_rl` is not a hand-written flag like `RL_mark`. It is a learned latent vector.

- `RL_mark` / `RLT gate`: decides when RL should take over.
- `z_rl`: compact representation given to the RL policy when RL is active.

They are related but different. The gate is a switch. The token is the information used by the RL controller.

## Train Stage 1

Default config uses the 5w checkpoint:

```bash
source /home/arts/anaconda3/etc/profile.d/conda.sh
conda activate ur3e_rlt

python scripts/rlt_token/train_rlt_stage1.py
```

Explicit command:

```bash
python scripts/rlt_token/train_rlt_stage1.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0610/checkpoints/050000/pretrained_model \
  --device cuda \
  --batch-size 8 \
  --steps 10000
```

Smoke test:

```bash
python scripts/rlt_token/train_rlt_stage1.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0610/checkpoints/020000/pretrained_model \
  --max-frames 8 \
  --max-steps 2 \
  --batch-size 2 \
  --num-workers 0 \
  --device cpu
```

The script reads the dataset path from the checkpoint's `train_config.json`.

## Outputs

Each run writes to `outputs/rlt_stage1/...`:

- `best.pt`: best Stage 1 encoder-decoder checkpoint by validation loss.
- `last.pt`: latest checkpoint.
- `stage1_config.json`: model dimensions, dataset path, action chunk timestamps.
- `summary.json`: short run summary.

The checkpoint contains:

```text
model_state
optimizer_state
model_config
embedding_dims
step
best_val
```

For later use, the important part is the encoder path:

```text
vlm_embedding + expert_embedding -> z_rl
```

The decoder is mainly for Stage 1 training.

## Optional Raw Token Extraction

`extract_rl_token.py` is still useful for debugging. It freezes SmolVLA and saves raw pooled VLA context tokens without training the Stage 1 encoder-decoder.

```bash
python scripts/rlt_token/extract_rl_token.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0610/checkpoints/050000/pretrained_model \
  --batch-size 8 \
  --device cuda
```

This produces `rl_token` with shape 960 by default. It is not the final compressed Stage 1 `z_rl`; it is the raw VLA context feature.

## How To Judge Stage 1

During training you should see:

```text
train_loss decreases
val_loss decreases or stabilizes
z_norm stays finite and does not explode
```

Do not expect Stage 1 loss to directly say whether the robot task succeeds. Stage 1 only says whether the token can preserve VLA internal information. Task success is evaluated later in the RL stage or rollout.

For the insertion task, ordinary VLA may approach the port but fail at the final contact/insertion stage. So before rollout, use offline diagnostics on demonstrations:

```bash
python scripts/rlt_token/eval_rlt_stage1.py \
  --checkpoint outputs/rlt_stage1/rlt_stage1_ur3e_smolvla_no_rotvec_20260609_231554_050000_20260610_215056/best.pt \
  --episodes 0 1 2 \
  --device cuda
```

This writes `outputs/rlt_stage1_eval/.../episode_xxxxxx/` with:

- `stage1_eval.npz`: raw arrays for `z`, `z_norm`, `dz_norm`, reconstruction losses, timestamps.
- `stage1_eval.png`: quick plot of token norm, token step size, losses, and 2D PCA trajectory.
- `summary.json`: numeric metrics for all evaluated episodes.

Interpretation:

- `z_norm` should be stable, not exploding over time.
- `dz_norm = ||z_t - z_{t-1}||` should mostly be smooth, with larger bumps around meaningful visual/contact/action changes.
- reconstruction loss should not spike randomly for long stretches.
- PCA trajectory should usually move continuously through the episode, not jump between unrelated clusters frame by frame.

This is not a success-rate test. It is a representation sanity check: the token should be smooth enough for a downstream RL policy, while still changing when the task state changes.

## Practical Defaults

For RTX 3070 / 8GB, start with:

```bash
--batch-size 4 --steps 10000
```

For 24GB GPU:

```bash
--batch-size 16 --steps 20000
```

If CUDA memory is tight, reduce:

```text
batch_size
hidden_dim
encoder_layers
```

The VLA is frozen, but extracting its embeddings is still the slow part.
