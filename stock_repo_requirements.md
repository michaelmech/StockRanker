# Stock ML Pipeline Repo — Overview and Requirements for Codex

## 1. Purpose

This repository exists to build a machine learning pipeline that predicts **cross-sectional stock rankings** so that a **market-neutral long/short strategy** can be effective in production.

This is not primarily a research sandbox. It is a full pipeline for:
- generating data,
- validating with time-aware CV,
- simulating realistic trading behavior,
- creating meta-labels,
- and executing live through Alpaca when a strategy is ready.

Research is still allowed, but mainly in service of improving the production pipeline. One example is using hyperparameter/search tooling such as Optuna to explore different stop losses, take profits, and horizons in order to shape better target variables.

---

## 2. Intended Users

This repository is for:
- the human operator only,
- and Codex agents working inside the repo.

It does **not** need to be optimized for a broad engineering team. It **does** need to be optimized for agent readability, modularity, and safe modification.

---

## 3. Primary Output

The repo should ultimately produce:
- a candidate **trading strategy object / package** that can be run in a notebook for inspection,
- along with strategy-generated **signals / positions** that can be reviewed,
- and then used for live execution if approved.

A strategy is considered ready for execution only when:
- it passes validation gates,
- notebook outputs make the case clearly,
- signals are in a form suitable for execution,
- and the strategy has survived enough iteration to be considered stable.

---

## 4. High-Level Pipeline

The intended pipeline is:

1. **Universe selection**  
   Select historically liquid stocks, likely with a historical volume filter.

2. **Data generation and rank-label creation**  
   Build the dataset and create cross-sectional rank labels.

3. **Feature engineering**  
   Generate predictive features per `(ticker, date)` observation.

4. **Primary model validation**  
   Validate using **time-aware cross-validation with purging**.

5. **Simulation for meta-labeling**  
   Use realistic simulation (targeting vectorbt-style workflows) to create meta-labels.

6. **Meta-model validation**  
   Validate the meta-classifier using time-aware CV.

7. **Iteration loop**  
   Iterate on targets, horizons, features, SL/TP logic, models, and portfolio settings until the best version is found.

8. **Deployment / execution**  
   Execute live through Alpaca once the strategy is validated.

---

## 5. Core Subsystems

All of the following are essential and should be treated as first-class subsystems:
- data generation
- feature engineering
- cross-validation / regime splitting
- feature selection
- modeling
- simulation
- meta-labeling
- execution

### Most fragile subsystems
These need extra protection during refactors and should be changed conservatively:
- **simulation**
- **meta-labeling**
- **execution**
- **cross-validation**

---

## 6. Core Design Principles

### 6.1 Strict time-series safety
This is the highest priority principle.

Requirements:
- no leakage across time,
- all validation must respect chronological order,
- purging must be enforced where relevant,
- regime-aware CV must remain safe,
- and anything that looks like it favors distant history too heavily should be treated skeptically.

### 6.2 Production realism in simulation
Simulation should be as close to live trading behavior as possible.

Requirements:
- liquidity constraints should matter,
- slippage and fees should be approximated,
- trade timing assumptions should be explicit,
- and simulation outputs must be trustworthy enough to support meta-labeling and deployment decisions.

### 6.3 Reproducibility
Reproducibility matters, but after time-safety and simulation realism.

Requirements:
- experiments should be reproducible where possible,
- outputs should be traceable to a specific configuration,
- and model / CV / simulation settings should be recoverable.

---

## 7. Validation Philosophy

### 7.1 Primary model metric
The core model metric should be:
- **cross-sectional Spearman**

This is the main success metric because the strategy is fundamentally a ranking problem.

### 7.2 What “good” means
The system should optimize for:
- strong out-of-sample ranking performance,
- robustness across folds,
- robustness across time regimes,
- and resistance to time-decay illusions.

In particular:
- models that look good mainly because they over-rely on older history should not be trusted,
- averages that mask recent degradation should be treated skeptically,
- and recent robustness matters more than flattering long-run averages.

### 7.3 Strategy promotion criteria
A strategy should only move toward deployment if it passes all of the following gates:
- strong CV performance,
- stable behavior across folds and regimes,
- realistic simulation performance,
- adversarial validation / drift checks.

---

## 8. Regimes and CV

### 8.1 CV direction
The repo should use **time-aware cross-validation with purging** as a first-class concept.

### 8.2 Preferred CV class
`RegimePurgedCV` should be treated as the **main CV class** conceptually, even if other CV classes exist in the repo.

### 8.3 Regime source
Regimes come from the **`create_cv_objects` workflow**, not from ad hoc manual handling during downstream model code.

That means the architecture should treat regime construction as part of the CV setup layer, and downstream model code should consume those objects rather than reinvent them.

---

## 9. Modeling Philosophy

### 9.1 Expected model family
The repo should primarily support:
- **gradient boosting models**

But the end state should be flexible enough to support:
- an **ensemble of different models**

### 9.2 Meta-labeling role
Meta-labeling is primarily a combination of:
- **trade filtering**, and
- **trade sizing / confidence weighting**

Depending on what fits the ranking/regression setup best.

That means the meta-model should sit on top of the primary ranking model and either:
- decide which opportunities are worth taking,
- or influence sizing based on confidence,
- or both.

---

## 10. Strategy Universe

### 10.1 Universe definition
The stock universe should depend primarily on **liquidity**.

At each datetime step, select stocks with the **most volume on that date** so that transaction-cost assumptions are more realistic.

### 10.2 Universe size
The intended live/simulated universe per rebalance date is roughly:
- **30 to 80 stocks**

This should be treated as a configurable operating range rather than a hardcoded constant.

### 10.3 Frequency
The strategy should support:
- **daily** operation,
- or **weekly** operation.

The exact cadence should be configurable.

### 10.4 Prediction horizon
The intended prediction horizon is:
- **1 day ahead**, or
- **1 week ahead**

depending on the chosen rebalance cadence.

---

## 11. Portfolio Construction

### 11.1 Neutrality philosophy
The strategy should aim to be **as neutral as possible**, while acknowledging that markets have a structural long bias over the long run.

The system should remain flexible enough to allow:
- configurable long/short counts,
- configurable long/short balance,
- and neutrality-oriented portfolio construction.

### 11.2 Number of positions
The number of long and short positions should be:
- specified via configurable arguments,
- and treated as something to simulate against,
- not hard-coded.

### 11.3 Position sizing modes
The strategy should support multiple sizing modes:
- **equal weight**
- **rank-based weight**
- **confidence-based weight** (including meta-label confidence)

These should be strategy-level configuration choices and simulation dimensions.

---

## 12. Trade Timing and Execution Assumptions

These assumptions are critical and should be preserved consistently across label generation, simulation, and live logic.

### 12.1 Entry timing
Assume positions are entered:
- **at the close of each day**

This should be the default conceptual timing model unless explicitly changed.

### 12.2 Stop loss / take profit timing
Stop losses and take profits:
- **do not trigger during after-hours trading**, and
- are only effective based on the **next regular session**, with overnight price movement reflected at the next open.

Practical interpretation:
- if price moves after hours beyond an SL or TP threshold,
- do **not** treat that as an immediate fill during after hours,
- instead reflect the impact starting from the **open of the following day**.

This is important because the simulation should not assume unrealistic intraday/overnight execution that production cannot replicate.

### 12.3 Exit philosophy
The strategy should primarily be:
- **rebalance-driven**

but also support:
- **stop-loss logic**
- **take-profit logic**

when that is consistent with the label-generation setup.

---

## 13. Label / Simulation / Execution Alignment

This is a hard architectural rule.

### 13.1 1:1 alignment requirement
The following should be kept as close to **1:1** as possible:
- label generation logic,
- simulation assumptions,
- and live trade-management logic.

If labels assume:
- a specific holding horizon,
- stop-loss behavior,
- take-profit behavior,
- trailing exits,
- rebalance timing,
- or close/open timing assumptions,

then simulation and execution should mirror that structure as closely as possible.

### 13.2 Why this matters
This is required so that:
- simulation remains trustworthy,
- meta-labels remain valid,
- and production behavior matches what the model actually learned.

Any mismatch between training labels and live execution logic should be considered a serious design problem.

---

## 14. Simulation Requirements

Simulation is one of the most sensitive parts of the repo.

### 14.1 Role of simulation
Simulation is not just for reporting returns. It is also used to:
- shape realistic expectations,
- support strategy promotion decisions,
- and create **meta-labels**.

### 14.2 Realism requirements
Simulation should approximate:
- slippage,
- fees / transaction costs,
- liquidity constraints,
- realistic tradability,
- and execution timing assumptions.

### 14.3 Liquidity gating rule
The system should include a liquidity / tradability gate.

Conceptually:
- if the **approximated simulated spread** or the **current spread** exceeds `stock_price * some_n_return_threshold`,
- the trade should **not** go through,
- and the stock should be treated as **illiquid for that decision point**.

This should be part of tradability filtering, not a cosmetic metric.

### 14.4 Simulation goal
Simulation should be realistic enough that notebook review can answer:
- is this actually tradable,
- are the assumptions credible,
- and is this close enough to production to support deployment?

---

## 15. Execution Layer

The execution layer exists to take a strategy that has passed the gates above and produce live trading actions through Alpaca.

Requirements:
- execution should consume strategy outputs cleanly,
- execution should not diverge conceptually from the simulation logic,
- and execution should remain consistent with label-generation assumptions wherever possible.

Execution is a core subsystem, not an afterthought.

---

## 16. Notebook-First Review Workflow

The intended operational workflow is:
- run strategy candidates in a notebook,
- inspect outputs,
- review whether validation/simulation results are convincing,
- and only then decide whether the strategy is ready to produce signals for live execution.

So while the repo supports execution, the immediate human-facing artifact should remain notebook-friendly and easy to inspect.

---

## 17. What Codex Should Preserve During Refactors

Codex should preserve the following invariants:

### Hard invariants
- strict chronological safety
- purged time-aware validation
- regime-aware CV as a first-class concept
- close alignment between labels, simulation, and execution
- realistic simulation assumptions
- liquidity-aware tradability filtering
- cross-sectional ranking as the main modeling objective
- cross-sectional Spearman as the main model metric

### Fragile areas to treat conservatively
- simulation
- meta-labeling
- execution
- cross-validation

### Refactor preference
Codex should optimize for:
- clearer architecture,
- explicit module boundaries,
- easier notebook usage,
- easier configuration of strategy variants,
- and safer experimentation,

without weakening the invariants above.

---

## 18. Non-Goals / Anti-Patterns

The repo should avoid drifting toward the following failure modes:
- leakage-prone validation shortcuts
- backtests that assume unrealistic fills
- label definitions that do not match live trade-management logic
- flattering long-run averages that hide recent decay
- strategy promotion based only on simulation aesthetics and not CV/drift robustness
- hard-coded portfolio choices that should really be configuration or simulation dimensions

---

## 19. Summary for Codex

If Codex needs the shortest possible brief, use this:

> Build and maintain a production-oriented ML pipeline for **cross-sectional stock ranking** in a **market-neutral long/short** framework. The repo must prioritize **strict time-series safety**, **purged regime-aware validation**, and **production-realistic simulation**. Labels, simulation behavior, and live execution logic should remain as close to **1:1** as possible. The key model metric is **cross-sectional Spearman**. The strategy universe should be a dynamic set of the most liquid stocks on each date, typically **30–80 names**, with configurable long/short counts and sizing modes. Simulation must account for slippage, transaction costs, and liquidity gating, and should be credible enough to support **meta-labeling** and deployment decisions. The most fragile and protected systems are **simulation, meta-labeling, execution, and cross-validation**.

