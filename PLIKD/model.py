import copy
import hashlib
import json
import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from transport import entropic_transport, sparse_transport, transport_distance


class PLIKD(nn.Module):
    def __init__(self, clip_model, tokenize, classnames, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.settings = dict(config.get("model", {}))
        self.loss_settings = dict(config.get("loss", {}))
        self.transport_settings = dict(config.get("transport", {}))
        self.clip_model = clip_model.float().eval().requires_grad_(False)
        self.tokenize = tokenize
        self.class_names = self._names(classnames)
        visual = self.clip_model.visual
        required = (
            "conv1",
            "class_embedding",
            "positional_embedding",
            "ln_pre",
            "transformer",
            "ln_post",
            "proj",
        )
        if any(not hasattr(visual, name) for name in required):
            raise ValueError("PLIKD requires the native OpenAI CLIP vision transformer")
        self.text_width = int(self.clip_model.token_embedding.weight.shape[1])
        self.joint_width = int(self.clip_model.text_projection.shape[1])
        self.visual_width = int(visual.conv1.weight.shape[0])
        self.sequence_length = int(self.clip_model.positional_embedding.shape[0])
        if self.text_width != self.joint_width:
            raise ValueError(
                "The configured CLIP must have matching joint and text embedding widths, as ViT-B/16 does"
            )
        self.visual_prompt_length = self._positive_integer("visual_prompt_length", 16)
        self.context_tokens = self._positive_integer("context_tokens", 16)
        hidden = self._positive_integer("projector_hidden", self.joint_width)
        self.text_support = self.settings.get("text_support", "token")
        if self.text_support not in {"token", "ensemble"}:
            raise ValueError("model.text_support must be token or ensemble")
        self.prompt_count = (
            1
            if self.text_support == "token"
            else self._positive_integer("text_prompts", 4)
        )
        if self.text_support == "ensemble" and self.prompt_count < 2:
            raise ValueError("A nondegenerate ensemble needs at least two text prompts")
        self.class_chunk_size = self._positive_integer("class_chunk_size", 16)
        self.visual_prompt = nn.Parameter(
            torch.empty(self.visual_prompt_length, self.visual_width)
        )
        self.context = nn.Parameter(
            torch.empty(self.prompt_count, self.context_tokens, self.text_width)
        )
        nn.init.normal_(self.visual_prompt, std=0.02)
        nn.init.normal_(self.context, std=0.02)
        self.projector = nn.Sequential(
            nn.Linear(self.joint_width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.joint_width),
        )
        self.temperature = float(self.loss_settings.get("temperature", 0.07))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("loss.temperature must be finite and positive")
        self._token_cache = {}
        self._fingerprint = None
        self.support_sizes(self.class_names)

    def _positive_integer(self, key, default):
        value = self.settings.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"model.{key} must be a positive integer")
        return value

    @staticmethod
    def _names(names):
        names = list(names)
        if not names or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError("Candidate class names must be nonempty strings")
        return names

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def _tokens(self, text):
        values = self.tokenize([text])
        if not isinstance(values, torch.Tensor) or values.ndim != 2 or len(values) != 1:
            raise ValueError("CLIP tokenize must return a two-dimensional token tensor")
        row = values[0].detach().cpu()
        end = int(row.argmax())
        if end < 2:
            raise ValueError("Class names must contain at least one content token")
        return int(row[0]), row[1:end].tolist(), int(row[end])

    def _specifications(self, names):
        key = tuple(names)
        if key not in self._token_cache:
            period = (
                self._tokens(".")[1]
                if self.settings.get("classname_period", True)
                else []
            )
            specifications = []
            for name in names:
                start, content, end = self._tokens(name.replace("_", " "))
                prefix = 2 + self.context_tokens
                sequence = (
                    [start] + [0] * (1 + self.context_tokens) + content + period + [end]
                )
                if len(sequence) > self.sequence_length:
                    raise ValueError(
                        f"Class prompt for {name!r} exceeds CLIP context length; truncation is disabled"
                    )
                specifications.append(
                    {
                        "tokens": sequence
                        + [0] * (self.sequence_length - len(sequence)),
                        "length": len(content),
                        "eot": len(sequence) - 1,
                        "class_start": prefix,
                    }
                )
            self._token_cache[key] = specifications
        return self._token_cache[key]

    def support_sizes(self, class_names=None):
        specifications = self._specifications(
            self._names(self.class_names if class_names is None else class_names)
        )
        return [
            self.prompt_count
            if self.text_support == "ensemble"
            else specification["length"] + 1
            for specification in specifications
        ]

    def _transform(self, transformer, values):
        if (
            self.training
            and torch.is_grad_enabled()
            and values.requires_grad
            and self.settings.get("activation_checkpointing", True)
        ):
            from torch.utils.checkpoint import checkpoint

            return checkpoint(transformer, values, use_reentrant=False)
        return transformer(values)

    def _visual(self, images, prompted=False):
        visual = self.clip_model.visual
        values = visual.conv1(images.to(dtype=visual.conv1.weight.dtype))
        values = values.flatten(2).transpose(1, 2)
        cls = visual.class_embedding.to(values.dtype).expand(len(values), 1, -1)
        values = torch.cat((cls, values), dim=1)
        if values.shape[1] != visual.positional_embedding.shape[0]:
            raise ValueError(
                "Image patch grid differs from the frozen CLIP positional grid"
            )
        values = values + visual.positional_embedding.to(values.dtype)
        if prompted:
            prompt = (
                self.visual_prompt.to(values.dtype)
                .unsqueeze(0)
                .expand(len(values), -1, -1)
            )
            values = torch.cat((values, prompt), dim=1)
        values = visual.ln_pre(values)
        values = self._transform(visual.transformer, values.transpose(0, 1)).transpose(
            0, 1
        )
        global_feature = visual.ln_post(values[:, 0]) @ visual.proj
        if prompted:
            return global_feature
        local_features = visual.ln_post(values[:, 1:]) @ visual.proj
        if local_features.shape[1] < 2:
            raise ValueError(
                "Fine-grained transport requires at least two visual support points"
            )
        return global_feature, local_features

    def _encode_support(self, z_r, context, specifications):
        batch = len(z_r)
        count = len(specifications)
        tokens = torch.tensor(
            [specification["tokens"] for specification in specifications],
            device=z_r.device,
        )
        fixed = self.clip_model.token_embedding(tokens)
        begin = 2 + self.context_tokens
        prefix = (
            fixed[:, :1]
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch, count, self.prompt_count, 1, self.text_width)
        )
        knowledge = z_r[:, None, None, None].expand(
            batch, count, self.prompt_count, 1, self.text_width
        )
        learned = context[None, None].expand(
            batch, count, self.prompt_count, self.context_tokens, self.text_width
        )
        suffix = (
            fixed[:, begin:]
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(
                batch,
                count,
                self.prompt_count,
                self.sequence_length - begin,
                self.text_width,
            )
        )
        values = torch.cat((prefix, knowledge, learned, suffix), dim=3)
        values = values + self.clip_model.positional_embedding[None, None, None]
        values = values.reshape(
            batch * count * self.prompt_count, self.sequence_length, self.text_width
        )
        values = self.clip_model.transformer(values.transpose(0, 1)).transpose(0, 1)
        values = self.clip_model.ln_final(values)
        if self.text_support == "ensemble":
            positions = torch.tensor(
                [specification["eot"] for specification in specifications],
                device=z_r.device,
            )
            positions = (
                positions[None, :, None]
                .expand(batch, count, self.prompt_count)
                .reshape(-1)
            )
            features = (
                values[torch.arange(len(values), device=z_r.device), positions]
                @ self.clip_model.text_projection
            )
            return features.reshape(batch, count, self.prompt_count, self.joint_width)
        specification = specifications[0]
        positions = list(range(begin, begin + specification["length"])) + [
            specification["eot"]
        ]
        features = values[:, positions] @ self.clip_model.text_projection
        return features.reshape(
            batch, count, self.prompt_count, len(positions), self.joint_width
        )[:, :, 0]

    def _support(self, z_r, specifications):
        if (
            self.training
            and torch.is_grad_enabled()
            and self.settings.get("activation_checkpointing", True)
        ):
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                lambda z, context: self._encode_support(z, context, specifications),
                z_r,
                self.context,
                use_reentrant=False,
            )
        return self._encode_support(z_r, self.context, specifications)

    def _logits(self, z_r, local_features, names):
        specifications = self._specifications(names)
        logits = [None] * len(names)
        plans = []
        diagnostics = {
            "row_residual": torch.zeros(len(z_r), device=z_r.device),
            "col_residual": torch.zeros(len(z_r), device=z_r.device),
            "converged_plans": torch.zeros(
                len(z_r), dtype=torch.long, device=z_r.device
            ),
            "total_plans": torch.zeros(len(z_r), dtype=torch.long, device=z_r.device),
            "iterations": torch.zeros(len(z_r), dtype=torch.long, device=z_r.device),
        }
        solver_name = self.transport_settings.get("solver", "sparse")
        solvers = {"sparse": sparse_transport, "entropic": entropic_transport}
        if solver_name not in solvers:
            raise ValueError("transport.solver must be sparse or entropic")
        epsilon = float(self.transport_settings.get("epsilon", 1.0))
        for start in range(0, len(names), self.class_chunk_size):
            groups = defaultdict(list)
            for index in range(start, min(start + self.class_chunk_size, len(names))):
                groups[
                    specifications[index]["length"]
                    if self.text_support == "token"
                    else 0
                ].append(index)
            for indices in groups.values():
                support = F.normalize(
                    self._support(z_r, [specifications[index] for index in indices]),
                    dim=-1,
                )
                cost = 1 - torch.einsum("bhd,bkzd->bkhz", local_features, support)
                plan, info = solvers[solver_name](
                    cost.detach(),
                    epsilon=epsilon,
                    max_iter=int(self.transport_settings.get("max_iter", 100)),
                    tol=float(self.transport_settings.get("tol", 1e-6)),
                    strict=bool(self.transport_settings.get("strict", False)),
                )
                row_residual = info["row_residual"].detach()
                col_residual = info["col_residual"].detach()
                tolerance = float(self.transport_settings.get("tol", 1e-6))
                diagnostics["row_residual"] = torch.maximum(
                    diagnostics["row_residual"], row_residual.amax(-1).float()
                )
                diagnostics["col_residual"] = torch.maximum(
                    diagnostics["col_residual"], col_residual.amax(-1).float()
                )
                diagnostics["converged_plans"] += (
                    torch.maximum(row_residual, col_residual) <= tolerance
                ).sum(-1)
                diagnostics["total_plans"] += len(indices)
                diagnostics["iterations"] = torch.maximum(
                    diagnostics["iterations"],
                    torch.full_like(diagnostics["iterations"], int(info["iterations"])),
                )
                regularized = bool(self.transport_settings.get("regularized", True))
                distance = transport_distance(
                    cost,
                    plan.detach(),
                    epsilon=epsilon,
                    regularized=regularized and solver_name == "sparse",
                )
                if regularized and solver_name == "entropic":
                    distance = distance + epsilon * torch.xlogy(
                        plan.detach(), plan.detach()
                    ).sum(dim=(-1, -2))
                for position, index in enumerate(indices):
                    logits[index] = (1 - distance[:, position]) / self.temperature
                if self.transport_settings.get("return_plans", False):
                    plans.append(
                        {
                            "class_indices": indices,
                            "plan": plan.detach(),
                            "diagnostics": info,
                        }
                    )
        return torch.stack(logits, dim=1), plans, diagnostics

    def forward(self, images, class_names=None, knowledge_features=None, labels=None):
        names = self._names(self.class_names if class_names is None else class_names)
        if images.ndim != 4 or images.shape[1] != 3 or not len(images):
            raise ValueError("images must be a nonempty Bx3xHxW tensor")
        with torch.no_grad():
            visual_global, visual_local = self._visual(images)
            visual_global = F.normalize(visual_global.float(), dim=-1)
            visual_local = F.normalize(visual_local.float(), dim=-1)
        z_r = self.projector(self._visual(images, prompted=True).float())
        logits, plans, diagnostics = self._logits(z_r, visual_local, names)
        losses = {}
        if knowledge_features is not None or labels is not None:
            targets = torch.arange(len(images), device=images.device)
            instance = F.normalize(z_r.float(), dim=-1)
            losses["visual"] = F.cross_entropy(
                instance @ visual_global.T / self.temperature, targets
            )
            if knowledge_features is not None:
                teacher = knowledge_features.detach().to(
                    device=z_r.device, dtype=torch.float32
                )
                if (
                    teacher.shape != z_r.shape
                    or not torch.isfinite(teacher).all()
                    or torch.any(teacher.norm(dim=-1) <= 0)
                ):
                    raise ValueError(
                        "Knowledge features must be finite nonzero Bxjoint_width vectors aligned with images"
                    )
                teacher = F.normalize(teacher, dim=-1)
                losses["knowledge"] = F.cross_entropy(
                    instance @ teacher.T / self.temperature, targets
                )
            losses["distill"] = (
                float(self.loss_settings.get("visual_weight", 0.3)) * losses["visual"]
            )
            if "knowledge" in losses:
                losses["distill"] = (
                    losses["distill"]
                    + float(self.loss_settings.get("knowledge_weight", 0.7))
                    * losses["knowledge"]
                )
            losses["total"] = losses["distill"]
            if labels is not None:
                if (
                    labels.ndim != 1
                    or len(labels) != len(images)
                    or labels.dtype
                    not in (
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    )
                ):
                    raise ValueError(
                        "Labels must be one integer local candidate index per image"
                    )
                labels = labels.to(device=logits.device, dtype=torch.long)
                if torch.any(labels < 0) or torch.any(labels >= len(names)):
                    raise ValueError(
                        "A label is outside the current candidate class set"
                    )
                classification = self.loss_settings.get("classification", "ce")
                if classification == "ce":
                    losses["classification"] = F.cross_entropy(logits, labels)
                elif classification == "literal_eq26":
                    losses["classification"] = -logits.softmax(-1)[
                        targets, labels
                    ].mean()
                else:
                    raise ValueError("loss.classification must be ce or literal_eq26")
                losses["total"] = (
                    losses["total"]
                    + float(self.loss_settings.get("classification_weight", 0.8))
                    * losses["classification"]
                )
        result = {
            "logits": logits,
            "z_r": z_r,
            "visual_global": visual_global,
            "visual_local": visual_local,
            "losses": losses,
            "transport_diagnostics": diagnostics,
        }
        if self.transport_settings.get("return_plans", False):
            result["plans"] = plans
        return result

    def trainable_state_dict(self):
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(self, state, strict=True):
        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if strict and set(state) != set(parameters):
            raise ValueError("Trainable checkpoint keys do not match the model")
        with torch.no_grad():
            for name, value in state.items():
                if (
                    name not in parameters
                    or not isinstance(value, torch.Tensor)
                    or value.shape != parameters[name].shape
                ):
                    raise ValueError(f"Invalid trainable checkpoint parameter: {name}")
                parameters[name].copy_(value.to(parameters[name]))

    def frozen_fingerprint(self):
        if self._fingerprint is None:
            result = hashlib.sha256()
            for name, tensor in sorted(self.clip_model.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                result.update(
                    json.dumps([name, list(value.shape), str(value.dtype)]).encode()
                )
                result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            self._fingerprint = result.hexdigest()
        return self._fingerprint

    def architecture(self):
        return {
            "visual_prompt_length": self.visual_prompt_length,
            "context_tokens": self.context_tokens,
            "projector_hidden": int(self.projector[0].out_features),
            "text_support": self.text_support,
            "text_prompts": self.prompt_count,
            "classname_period": bool(self.settings.get("classname_period", True)),
            "visual_width": self.visual_width,
            "joint_width": self.joint_width,
            "text_width": self.text_width,
            "sequence_length": self.sequence_length,
        }
