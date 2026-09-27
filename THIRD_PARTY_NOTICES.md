# Source and adaptation

The estimation code adapts Martí Mestieri's `simulation_countrypanel_web.do` and `montecarlo_web.do`, available through the estimation-code link on [his research page](https://mestieri.github.io/). The supplied Stata files specify [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

Python adaptation and modifications: **Wenzhuo Wang**.

| Python file | Source and main changes |
|---|---|
| `estimation/simulate.py` | Panel and measurement-error design from `simulation_countrypanel_web.do`; uses the note's parameter convention, NumPy random draws, and separate observed/truth tables. |
| `estimation/estimators.py` | Seven specifications from the same Stata file; adds numerical checks and explicit parameter-identification labels. |
| `estimation/run_montecarlo.py` | Repeated-experiment workflow from `montecarlo_web.do`; adds configurable runs, CSV/JSON output, failure counts, and bias/RMSE summaries. |

These three adaptations retain CC BY-SA 4.0. Original Stata files are not bundled, and attribution does not imply endorsement. The original model and tests use MIT; the study note uses CC BY 4.0. See [LICENSE](LICENSE) for the repository's licensing terms.
