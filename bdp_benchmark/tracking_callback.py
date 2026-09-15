"""Rollout averages of tracker diagnostics, using the existing SB3 logger."""

from stable_baselines3.common.callbacks import BaseCallback

from .tracking_diagnostics import TrackingRecorder


class TrackingCallback(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.recorder = None
        self.totals = {}
        self.count = 0

    def _on_training_start(self):
        self.recorder = TrackingRecorder(self.path)

    def _on_rollout_start(self):
        self.totals = {}
        self.count = 0

    def _on_step(self):
        for info in self.locals.get("infos", ()):
            metrics = info.get("tracking")
            if metrics is not None:
                self.count += 1
                for key, value in metrics.items():
                    self.totals[key] = self.totals.get(key, 0.0) + value
        return True

    def _on_rollout_end(self):
        if not self.count:
            return
        means = {key: value / self.count for key, value in self.totals.items()}
        for key, value in means.items():
            self.logger.record(f"tracking/{key}", value)
        self.recorder.write({"timesteps": self.num_timesteps, "samples": self.count, **means})
        self.recorder.stream.flush()

    def close(self):
        if self.recorder is not None:
            self.recorder.close()

    def _on_training_end(self):
        self.close()
