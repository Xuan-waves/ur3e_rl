from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class HILTransition:
    z_rl: np.ndarray
    state: np.ndarray
    ref_action: np.ndarray
    action: np.ndarray
    reward: float
    next_z_rl: np.ndarray
    next_state: np.ndarray
    next_ref_action: np.ndarray
    done: float
    is_intervention: float

    def to_npz_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "z_rl": self.z_rl,
            "state": self.state,
            "ref_action": self.ref_action,
            "action": self.action,
            "reward": float(self.reward),
            "next_z_rl": self.next_z_rl,
            "next_state": self.next_state,
            "next_ref_action": self.next_ref_action,
            "done": float(self.done),
            "is_intervention": float(self.is_intervention),
        }


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._data: deque[HILTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._data)

    def insert(self, transition: HILTransition) -> None:
        self._data.append(transition)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if not self._data:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        items = random.choices(tuple(self._data), k=int(batch_size))
        out: dict[str, list[Any]] = {
            "z_rl": [],
            "state": [],
            "ref_action": [],
            "action": [],
            "reward": [],
            "next_z_rl": [],
            "next_state": [],
            "next_ref_action": [],
            "done": [],
            "is_intervention": [],
        }
        for item in items:
            out["z_rl"].append(item.z_rl)
            out["state"].append(item.state)
            out["ref_action"].append(item.ref_action)
            out["action"].append(item.action)
            out["reward"].append([item.reward])
            out["next_z_rl"].append(item.next_z_rl)
            out["next_state"].append(item.next_state)
            out["next_ref_action"].append(item.next_ref_action)
            out["done"].append([item.done])
            out["is_intervention"].append([item.is_intervention])
        return {key: torch.as_tensor(np.asarray(value), dtype=torch.float32) for key, value in out.items()}

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._data:
            np.savez_compressed(path, size=np.asarray([0], dtype=np.int64))
            return
        keys = list(self._data[0].to_npz_dict().keys())
        arrays = {
            key: np.asarray([item.to_npz_dict()[key] for item in self._data], dtype=np.float32)
            for key in keys
        }
        arrays["size"] = np.asarray([len(self._data)], dtype=np.int64)
        np.savez_compressed(path, **arrays)

    def load_npz(self, path: Path) -> int:
        data = np.load(path)
        size = int(data["size"][0]) if "size" in data.files else int(len(data["reward"]))
        self._data.clear()
        for i in range(size):
            self.insert(
                HILTransition(
                    z_rl=np.asarray(data["z_rl"][i], dtype=np.float32),
                    state=np.asarray(data["state"][i], dtype=np.float32),
                    ref_action=np.asarray(data["ref_action"][i], dtype=np.float32),
                    action=np.asarray(data["action"][i], dtype=np.float32),
                    reward=float(data["reward"][i]),
                    next_z_rl=np.asarray(data["next_z_rl"][i], dtype=np.float32),
                    next_state=np.asarray(data["next_state"][i], dtype=np.float32),
                    next_ref_action=np.asarray(data["next_ref_action"][i], dtype=np.float32),
                    done=float(data["done"][i]),
                    is_intervention=float(data["is_intervention"][i]),
                )
            )
        return len(self._data)


class RLTActor(nn.Module):
    """Residual actor: final_action_chunk = ref_action_chunk + tanh(delta) * action_scale."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, action_scale: np.ndarray) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32).reshape(1, action_dim))

    def forward(self, z_rl: torch.Tensor, state: torch.Tensor, ref_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_rl, state, ref_action], dim=-1)
        delta = torch.tanh(self.net(x)) * self.action_scale
        return ref_action + delta


class TwinQCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()

        def make_q() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        ref_action: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_rl, state, ref_action, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        ref_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z_rl, state, ref_action, action], dim=-1)
        return self.q1(x)


@dataclass(slots=True)
class HILSERLConfig:
    z_dim: int = 256
    state_dim: int = 4
    action_dim: int = 4
    train_action_dim: int = 3
    action_chunk_steps: int = 10
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    action_delta_scale_xyz: float = 0.015
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.98
    tau: float = 0.005
    batch_size: int = 128
    replay_demo_ratio: float = 0.5
    bc_weight: float = 0.2
    target_noise_xyz: float = 0.002
    target_noise_clip_xyz: float = 0.006
    train_after: int = 64
    updates_per_step: int = 1
    policy_delay: int = 2
    grad_clip_norm: float = 10.0
    action_low: tuple[float, ...] = (-1.0, -1.0, 0.0)
    action_high: tuple[float, ...] = (1.0, 1.0, 1.0)


class HILSERLTrainer:
    def __init__(self, cfg: HILSERLConfig, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.cfg = cfg
        self.device = torch.device(device)
        self.train_action_dim = int(cfg.train_action_dim)
        if self.train_action_dim < 1 or self.train_action_dim > int(cfg.action_dim):
            raise ValueError(f"train_action_dim must be in [1, action_dim], got {self.train_action_dim}")
        self.action_chunk_steps = max(1, int(cfg.action_chunk_steps))
        self.flat_action_dim = self.action_chunk_steps * self.train_action_dim
        action_scale = np.full((self.flat_action_dim,), cfg.action_delta_scale_xyz, dtype=np.float32)
        actor_in = cfg.z_dim + cfg.state_dim + self.flat_action_dim
        critic_in = cfg.z_dim + cfg.state_dim + self.flat_action_dim + self.flat_action_dim
        self.actor = RLTActor(actor_in, self.flat_action_dim, cfg.actor_hidden_dim, action_scale).to(self.device)
        self.actor_target = RLTActor(actor_in, self.flat_action_dim, cfg.actor_hidden_dim, action_scale).to(self.device)
        self.critic = TwinQCritic(critic_in, cfg.critic_hidden_dim).to(self.device)
        self.critic_target = TwinQCritic(critic_in, cfg.critic_hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        self.update_step = 0

        low = np.asarray(cfg.action_low[: self.train_action_dim], dtype=np.float32)
        high = np.asarray(cfg.action_high[: self.train_action_dim], dtype=np.float32)
        self.action_low = torch.as_tensor(np.tile(low, self.action_chunk_steps), dtype=torch.float32, device=self.device).reshape(
            1, self.flat_action_dim
        )
        self.action_high = torch.as_tensor(
            np.tile(high, self.action_chunk_steps),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, self.flat_action_dim)

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _flat_train_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions[:, : self.action_chunk_steps, : self.train_action_dim]
        if actions.shape[1] < self.action_chunk_steps:
            pad = actions[:, -1:, :].expand(-1, self.action_chunk_steps - actions.shape[1], -1)
            actions = torch.cat([actions, pad], dim=1)
        return actions.reshape(actions.shape[0], self.flat_action_dim)

    def _numpy_chunk(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] < self.action_chunk_steps:
            pad = np.repeat(arr[-1:], self.action_chunk_steps - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        return arr[: self.action_chunk_steps].copy()

    def act_chunk(self, z_rl: np.ndarray, state: np.ndarray, ref_action_chunk: np.ndarray) -> np.ndarray:
        self.actor.eval()
        ref_chunk_np = self._numpy_chunk(ref_action_chunk)
        with torch.inference_mode():
            z = torch.as_tensor(z_rl, dtype=torch.float32, device=self.device).reshape(1, -1)
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            ref_full = torch.as_tensor(ref_chunk_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            ref = self._flat_train_actions(ref_full)
            flat_action = self.actor(z, s, ref).clamp(self.action_low, self.action_high)
        out = ref_chunk_np.copy()
        out[:, : self.train_action_dim] = (
            flat_action.detach().cpu().numpy().astype(np.float32).reshape(self.action_chunk_steps, self.train_action_dim)
        )
        return out

    def act(self, z_rl: np.ndarray, state: np.ndarray, ref_action: np.ndarray) -> np.ndarray:
        return self.act_chunk(z_rl, state, ref_action)[0]

    def update(self, online: ReplayBuffer, intervention: ReplayBuffer) -> dict[str, float] | None:
        if len(online) < self.cfg.train_after:
            return None
        batch_size = int(self.cfg.batch_size)
        batch = online.sample(batch_size)
        batch = self._to_device(batch)
        batch["ref_action"] = self._flat_train_actions(batch["ref_action"])
        batch["action"] = self._flat_train_actions(batch["action"])
        batch["next_ref_action"] = self._flat_train_actions(batch["next_ref_action"])

        bc_batch = None
        if len(intervention) > 0 and self.cfg.bc_weight > 0.0:
            bc_bs = int(round(batch_size * float(self.cfg.replay_demo_ratio)))
            bc_bs = max(1, min(bc_bs, batch_size))
            bc_batch = self._to_device(intervention.sample(bc_bs))
            bc_batch["ref_action"] = self._flat_train_actions(bc_batch["ref_action"])
            bc_batch["action"] = self._flat_train_actions(bc_batch["action"])

        with torch.no_grad():
            next_action = self.actor_target(batch["next_z_rl"], batch["next_state"], batch["next_ref_action"])
            if self.cfg.target_noise_xyz > 0.0:
                noise = torch.randn_like(next_action) * float(self.cfg.target_noise_xyz)
                noise = noise.clamp(-float(self.cfg.target_noise_clip_xyz), float(self.cfg.target_noise_clip_xyz))
                next_action = next_action + noise
            next_action = next_action.clamp(self.action_low, self.action_high)
            tq1, tq2 = self.critic_target(
                batch["next_z_rl"],
                batch["next_state"],
                batch["next_ref_action"],
                next_action,
            )
            target_q = batch["reward"] + (1.0 - batch["done"]) * self.cfg.gamma * torch.minimum(tq1, tq2)
            target_q = target_q.detach()

        q1, q2 = self.critic(batch["z_rl"], batch["state"], batch["ref_action"], batch["action"])
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
        self.critic_opt.step()

        actor_loss_value = torch.tensor(0.0, device=self.device)
        bc_loss_value = torch.tensor(0.0, device=self.device)
        q_loss_value = torch.tensor(0.0, device=self.device)
        actor_updated = False
        if self.update_step % max(1, int(self.cfg.policy_delay)) == 0:
            actor_updated = True
            actor_action = self.actor(batch["z_rl"], batch["state"], batch["ref_action"])
            actor_action = actor_action.clamp(self.action_low, self.action_high)
            for param in self.critic.parameters():
                param.requires_grad_(False)
            try:
                q_loss = -self.critic.q1_value(batch["z_rl"], batch["state"], batch["ref_action"], actor_action).mean()

                if bc_batch is not None:
                    bc_actor_action = self.actor(bc_batch["z_rl"], bc_batch["state"], bc_batch["ref_action"])
                    bc_actor_action = bc_actor_action.clamp(self.action_low, self.action_high)
                    bc_loss = F.mse_loss(bc_actor_action, bc_batch["action"])
                else:
                    bc_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                actor_loss = q_loss + self.cfg.bc_weight * bc_loss
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
                self.actor_opt.step()
            finally:
                for param in self.critic.parameters():
                    param.requires_grad_(True)
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
            actor_loss_value = actor_loss.detach()
            bc_loss_value = bc_loss.detach()
            q_loss_value = q_loss.detach()

        self.update_step += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss_value.detach().cpu()),
            "bc_loss": float(bc_loss_value.detach().cpu()),
            "actor_q_loss": float(q_loss_value.detach().cpu()),
            "q_mean": float(q1.detach().mean().cpu()),
            "actor_updated": float(actor_updated),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        tau = float(self.cfg.tau)
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def save(self, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "update_step": self.update_step,
                "config": asdict(self.cfg),
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.update_step = int(ckpt.get("update_step", 0))
