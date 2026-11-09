# Journal release checklist

This checklist records work that should be completed before the repository is made public and archived.

- [x] Replace legacy sparse-autoencoder README.
- [x] Document final datasets, subject inclusion, and event mappings.
- [x] Identify final journal experiment configuration in `oddball_eeg.config`.
- [x] Preserve full development environment and add a portable environment.
- [x] Document notebook roles and the regeneration workflow.
- [x] Add software license and citation metadata.
- [x] Package reusable code (`src/oddball_eeg`), organise notebooks, strip outputs.
- [x] Add unit/smoke tests and continuous integration.
- [x] Verify the portable environment from a clean checkout.
- [x] Verify the exact OpenNeuro snapshot/version identifiers and dataset citations.
- [x] Re-run at least dataset loading and one representative analysis path on the real data after the package refactor (e.g. notebook 02 for ds003061) and compare against the thesis tables.
- [x] Regenerate any lightweight derived result files intended for public distribution.
- [x] Confirm CRediT/authorship and public-release approval with all authors.
- [x] Create a versioned GitHub release (v1.0.0).
- [x] Archive the release with Zenodo (version DOI 10.5281/zenodo.24070051; concept DOI 10.5281/zenodo.24070050) and add the DOI to `CITATION.cff` and the README.
- [x] Add the version DOI to the manuscript and Supplementary Material.
- [x] Add the article DOI to `CITATION.cff` and the README on publication.
