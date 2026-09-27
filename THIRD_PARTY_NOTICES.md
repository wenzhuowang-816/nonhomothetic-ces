# Attribution and adaptation record

Original source: Martí Mestieri, `simulation_countrypanel_web.do` and `montecarlo_web.do`, available via the **estimation code** link on [his research page](https://mestieri.github.io/).

Both supplied Stata files identify Martí Mestieri as their author and specify [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The [full legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode) governs those adaptations. Original Stata source files are not bundled here.

| Python file | Source and changes |
|---|---|
| `estimation/simulate.py` | Adapts the panel and measurement-error design from `simulation_countrypanel_web.do`; uses the note's parameter convention, a local NumPy RNG, and separate observed/truth tables. |
| `estimation/estimators.py` | Adapts its seven estimation specifications; adds numerical safeguards and explicit parameter-identification labels. |
| `estimation/run_montecarlo.py` | Adapts the repeated-experiment workflow of `montecarlo_web.do`; adds configurable CLI, CSV/JSON outputs, failure counts, bias/RMSE, and source hashes. |

These three files remain under CC BY-SA 4.0. Python adaptation and modifications: Wenzhuo Wang. Attribution does not imply endorsement by Mestieri, his institutions, or the paper's other authors.

The model implementation and tests are separately written from the mathematical definitions and are covered by the original-work MIT terms in `LICENSE`. The study note is covered by the CC BY 4.0 terms already specified there; [full legal code](https://creativecommons.org/licenses/by/4.0/legalcode).

The dependency packages retain their own licenses. This repository does not redistribute their source code.
