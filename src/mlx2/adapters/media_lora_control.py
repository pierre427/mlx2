"""Adapter-owned media LoRA lifecycle, serialized with generation."""

from __future__ import annotations

import functools
import threading

from mlx2.runtime.media_lora import inspect_media_lora, install_media_lora


def serialized(method):
    @functools.wraps(method)
    def run(self, *args, **kwargs):
        with self._media_lock:
            return method(self, *args, **kwargs)

    return run


class MediaLoRAControl:
    def _init_lora(self, family, revision):
        self._media_lock = threading.RLock()
        self._lora_family = family
        self._lora_revision = revision
        self._lora = None
        self._lora_sessions = []
        self._lora_epoch = 0
        self._lora_observed = False

    @property
    def lora_base_fingerprint(self):
        return getattr(self, "_lora_base_identity", self.artifact.fingerprint)

    @serialized
    def load_lora(self, path):
        if self._lora is not None:
            raise RuntimeError("unload current media LoRA before loading another")
        artifact = inspect_media_lora(
            path,
            family=self._lora_family,
            base_fingerprint=self.lora_base_fingerprint,
            backend_revision=self._lora_revision,
        )
        sessions = []
        try:
            # Backends expose the actual transformer; validation cannot stop at a wrapper.
            for transformer in self._lora_models():
                sessions.append(install_media_lora(transformer, artifact))
        except BaseException:
            for session in reversed(sessions):
                session.restore()
            raise
        self._lora, self._lora_sessions = artifact, sessions
        self._lora_epoch += 1
        self._lora_observed = False
        return self.lora_receipt()

    @serialized
    def train_lora(self, *, keys, examples, validation_examples, output, **options):
        if self._lora is not None:
            raise RuntimeError("unload current media LoRA before training")
        from mlx2.runtime.media_lora_training import train_media_lora

        models = self._lora_models()
        if len(models) != 1:
            raise RuntimeError(
                "training requires one backend transformer; use train_media_lora for an explicit model"
            )
        return train_media_lora(
            models[0],
            family=self._lora_family,
            base_fingerprint=self.lora_base_fingerprint,
            backend_revision=self._lora_revision,
            keys=keys,
            examples=examples,
            validation_examples=validation_examples,
            output=output,
            **options,
        )

    @serialized
    def unload_lora(self):
        for session in self._lora_sessions:
            session.check_restore()
        for session in reversed(self._lora_sessions):
            session.restore()
        self._lora_sessions = []
        self._lora = None
        self._lora_epoch += 1
        self._lora_observed = False
        return self.lora_receipt()

    def _attach_lora(self, transformer):
        if self._lora is not None:
            session = install_media_lora(transformer, self._lora)
            self._lora_sessions.append(session)

    def _mark_lora_used(self):
        if self._lora is not None:
            self._lora_observed = True

    @serialized
    def lora_receipt(self):
        return {
            "family": self._lora_family,
            "base_fingerprint": self.lora_base_fingerprint,
            "backend_revision": self._lora_revision,
            "adapter_fingerprint": self._lora.fingerprint if self._lora else None,
            "state_epoch": self._lora_epoch,
            "implemented": True,
            "trained": self._lora.config["training"]["trained"]
            if self._lora
            else False,
            "qualified": False,
            "selected": self._lora is not None,
            "observed_used": self._lora_observed,
        }
