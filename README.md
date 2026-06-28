# LLM-DRRN TextWorld Agent

A hybrid reinforcement learning agent that fuses a **frozen LLM** (Llama 3) with a
**Deep Reinforcement Relevance Network (DRRN)** to solve TextWorld text adventures.

The LLM acts as a linguistic intuition engine that filters the action space down to K
high-value candidates per step. The DRRN then scores those candidates using a learned
dot-product Q-function and selects the best one via biased ε-greedy exploration.

---

## Architecture

```
TextWorld Env
     │ obs, reward
     ▼
Action-History Buffer      (rolling window of 3 transitions → augmented state S_t)
     │ S_t
     ├──────────────────────────────────────────────────────────┐
     ▼                                                          ▼
Frozen LLM (Llama 3)                                   DRRN State Encoder
     │ K candidates + probs P                                   │ h_s
     ▼                                                          │
DRRN Action Encoder ◄──────────────────────────────────────────┘
     │ h_{a_k}
     ▼
Dot Product  →  Q(S, a_k) = h_s · h_a
     │
LLM-Biased ε-Greedy
  Exploit: argmax Q
  Explore: sample from P (LLM distribution)
     │
     ▼
Action → Environment
```

### The 5 modules

| Module | File | Role |
|--------|------|------|
| Temporal State Manager | `src/agent/state_manager.py` | Rolling `deque(maxlen=3)` of `(action, obs)` pairs |
| Intuition Engine | `src/agent/llm_engine.py` | Frozen LLM → K candidates + probability distribution P |
| Evaluator (DRRN) | `src/agent/drrn.py` | Frozen sentence-transformer + trainable projection + dot product |
| Execution Policy | `src/agent/policy.py` | Biased ε-greedy (exploit=argmax Q, explore=sample P) |
| Optimization Loop | `src/agent/agent.py` + `src/training/trainer.py` | TD loss + target network + AdamW |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

TextWorld also requires Frotz (a Z-machine interpreter):
```bash
# Ubuntu/Debian
sudo apt-get install -y frotz

# macOS
brew install frotz

# Windows: use WSL, or download frotz binary and add to PATH
```

### 2. Choose your LLM backend

#### Option A — Ollama (recommended, zero GPU overhead for LLM)

```bash
# Install Ollama: https://ollama.ai/
ollama pull llama3
# Ollama runs as a local server on port 11434
```

`config.yaml`:
```yaml
llm:
  backend: "ollama"
  model_name: "llama3"
```

#### Option B — HuggingFace (requires ~8 GB VRAM for Llama 3 8B with 4-bit quant)

```bash
huggingface-cli login   # need access to meta-llama/Meta-Llama-3-8B-Instruct
```

`config.yaml`:
```yaml
llm:
  backend: "huggingface"
  model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
  hf_load_in_4bit: true
```

#### Option C — Mock (no LLM, uniform random from admissible_commands)

```yaml
llm:
  backend: "mock"
```

Use this for quick sanity checks or ablation studies.

### 3. Generate a game

```bash
python generate_game.py --output games/ --seed 42 --rooms 5
```

Update `config.yaml` with the printed game path.

### 4. Train

```bash
python train.py
```

Useful flags:
```bash
python train.py --backend mock --episodes 100    # fast ablation
python train.py --resume checkpoints/best.pt     # resume training
python train.py --game games/my_game.ulx         # custom game
```

### 5. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt --episodes 20
```

---

## Key design decisions

### Why dot product instead of MLP for Q?

The original DRRN (He et al., 2016) uses an MLP to combine state and action
vectors. Using a dot product `Q = h_s · h_a` after a shared projection is:
- Lighter: no extra MLP parameters
- More stable: avoids vanishing gradients in a deep head
- Principled: equivalent to a metric-learning objective

### Why biased exploration?

Standard ε-greedy picks any action uniformly at random when exploring.
LLM-biased exploration samples from the LLM's probability distribution P,
so even random exploration is steered toward linguistically plausible actions.
This dramatically reduces the amount of nonsensical actions the agent tries
early in training.

### Why freeze the sentence transformer?

Fine-tuning a sentence transformer during RL is unstable — the embedding space
shifts and invalidates stored replay transitions. The projection layer (128 dims)
is tiny enough to train stably while the frozen encoder provides a rich
semantic prior.

### Why store next_candidates in the replay buffer?

Because the action space is dynamic (different rooms have different valid
commands), the standard `S, A, R, S'` tuple is insufficient. We need to know
**which actions were available at S'** to compute `max_a' Q(S', a')` in the
TD target. This is a critical implementation difference from standard DQN.

---

## Repository structure

```
llm-drrn-textworld/
├── config.yaml                     # all hyperparameters
├── requirements.txt
├── generate_game.py                # TextWorld game generator
├── train.py                        # training entry point
├── evaluate.py                     # evaluation entry point
├── games/                          # generated .ulx files go here
└── src/
    ├── environment/
    │   └── textworld_env.py        # gym wrapper requesting admissible_commands
    └── agent/
        ├── state_manager.py        # Module 1: rolling history buffer
        ├── llm_engine.py           # Module 2: LLM candidate generator
        ├── drrn.py                 # Module 3: DRRN Q-network
        ├── policy.py               # Module 4: biased ε-greedy
        ├── replay_buffer.py        # stores (S, A, R, S', Cands') transitions
        └── agent.py                # orchestrates all modules + TD update
    └── training/
        └── trainer.py              # episode loop + checkpointing
```

---

## References

- He, J. et al. (2016). **Deep Reinforcement Learning with a Natural Language Action Space**. ACL.  
- Côté, M.-A. et al. (2019). **TextWorld: A Learning Environment Sandbox for Developing and Evaluating Text-Based Game Playing Agents**. CGW@AAAI.  
- Murugesan, K. et al. (2021). **Text-based RL Agents with Commonsense Knowledge**. AAAI. (IBM TWC repo)  
- RLSS 2019 Final Project: `yfletberliac/rlss-2019` on GitHub.
