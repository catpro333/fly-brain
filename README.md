# fly-brain

A tiny, biologically-inspired model of a fly brain built in Python.

What it does:
- Models sensory channels for smell, vision, and threat
- Tracks memory and decision-making
- Chooses a behavior such as forage, explore, or escape
- Runs as a simple command-line simulation

Quick start:

```bash
python fly_brain.py
```

The demo prints a few simulated sensory states and the resulting fly behavior.

Example output:

```text
time=0 odor=0.80 light=0.10 danger=0.20
  olfactory: 0.81 visual: 0.17 decision: 0.65 action: forage
```

This is a toy model intended to be extensible, not a full neural reconstruction of a real fly brain.
