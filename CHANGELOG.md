# Changes in the reviewed version

- Reject nonfinite model outputs and mark overflowing normalized estimates as failures.
- Add repository-root import configuration for pytest and VS Code test discovery.
- Give CLI/F5 runs fresh timestamped output folders by default; keep explicit-output overwrite protection.
- Record source-file hashes alongside experiment configuration and software versions.
- Track shared VS Code configurations and the two curated result snapshots.
- Add model-identity, simulation, Monte Carlo accounting, output-safety, and numerical-boundary tests.
- Correct method 5 documentation, document normalization and estimator limitations, and clarify source attribution.
- Preserve the supplied study-note PDF and both existing result directories byte-for-byte. Generated Python/test caches are omitted from the reviewed archive.

The estimator equations, default model parameters, and original licensing choices are retained. This review does not claim a direct Stata execution comparison.
