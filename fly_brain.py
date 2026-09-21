from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + __import__("math").exp(-x))


@dataclass
class BrainRegion:
    name: str
    activation: float = 0.0
    bias: float = 0.0
    gain: float = 1.0
    memory: float = 0.0

    def update(self, signal: float) -> float:
        self.activation = sigmoid((signal + self.bias) * self.gain)
        self.memory = 0.7 * self.memory + 0.3 * self.activation
        return self.activation


class FlyBrain:
    """A compact, biologically-inspired model of a fly's sensory and behavioral loops."""

    def __init__(self) -> None:
        self.olfactory = BrainRegion("olfactory", bias=0.5, gain=2.0)
        self.visual = BrainRegion("visual", bias=-0.2, gain=1.8)
        self.threat = BrainRegion("threat", bias=0.8, gain=2.5)
        self.memory = BrainRegion("memory", bias=0.0, gain=2.0)
        self.decision = BrainRegion("decision", bias=0.0, gain=1.5)
        self.motor = BrainRegion("motor", bias=0.0, gain=1.7)

    def step(self, odor: float, light: float, danger: float) -> Dict[str, float | str]:
        odor_signal = max(0.0, min(1.0, odor))
        light_signal = max(0.0, min(1.0, light))
        danger_signal = max(0.0, min(1.0, danger))

        olfactory = self.olfactory.update(odor_signal)
        visual = self.visual.update(light_signal)
        threat = self.threat.update(danger_signal)

        # Memory integrates recent sensory experience so behavior is not purely reactive.
        memory_signal = 0.45 * olfactory + 0.35 * visual + 0.20 * threat
        memory = self.memory.update(memory_signal)

        # Decision layer balances attraction, threat, and learned context.
        decision_signal = 0.55 * olfactory + 0.20 * visual + 0.25 * memory - 0.80 * threat
        decision = self.decision.update(decision_signal)

        action = self._select_action(decision, threat, memory)
        motor_signal = self._motor_signal(action)
        motor = self.motor.update(motor_signal)

        return {
            "olfactory": round(olfactory, 3),
            "visual": round(visual, 3),
            "threat": round(threat, 3),
            "memory": round(memory, 3),
            "decision": round(decision, 3),
            "motor": round(motor, 3),
            "action": action,
        }

    def _select_action(self, decision: float, threat: float, memory: float) -> str:
        if threat > 0.7:
            return "escape"
        if decision > 0.68 and memory > 0.45:
            return "forage"
        if decision < 0.35:
            return "explore"
        return "track"

    def _motor_signal(self, action: str) -> float:
        action_map = {
            "escape": 0.95,
            "forage": 0.80,
            "track": 0.55,
            "explore": 0.35,
        }
        return action_map.get(action, 0.5)


def demo() -> None:
    brain = FlyBrain()
    scenarios = [
        (0.8, 0.1, 0.2),
        (0.6, 0.9, 0.2),
        (0.2, 0.4, 0.9),
        (0.1, 0.7, 0.3),
    ]

    for index, (odor, light, danger) in enumerate(scenarios):
        result = brain.step(odor, light, danger)
        print(f"time={index} odor={odor} light={light} danger={danger}")
        print(
            "  olfactory={olfactory} visual={visual} threat={threat} "
            "memory={memory} decision={decision} action={action}".format(**result)
        )


if __name__ == "__main__":
    demo()
