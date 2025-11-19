# Repository Guidelines

## Project Structure & Module Organization
Elekta/ and Varian Truebeam/ contain the MATLAB apps (`*.mlapp`) plus their script equivalents (`*.m`) for static, dynamic, and treatment-interruption QA workflows. Lyrebird Data/ hosts sample centroid files and trajectory logs (e.g., `Lyrebird Data/Centroid_248687_BeamID_3.1_3.2.txt`) used in the static test walkthrough. Robot traces/ provides canned 6 DoF motion traces for dynamic validation, while the project root stores reference papers and TODOs. Keep subfolder layouts intact so the MATLAB apps can load relative resources without extra configuration.

## Build, Test, and Development Commands
- `matlab -batch "open('Elekta/App_Static_loc.mlapp')"` launches the static localisation app in the Elekta workspace.
- `matlab -batch "run('Elekta/Staticloc.m')"` executes the script version headlessly with default Lyrebird sample inputs.
- Swap `Elekta` for `Varian Truebeam` to exercise the alternative couch-shift convention. Use `matlab -batch "help Staticloc"` for quick signature checks before wiring new datasets.

## Coding Style & Naming Conventions
Follow existing MATLAB conventions: 4-space indentation, camelCase for variables (`seedCoords`), and UpperCamelCase for functions (`Staticloc`). Mirror filenames across app/script pairs (`App_*` ↔ `*.m`) so clinicians can map GUI to batch workflows. Keep hard-coded paths relative (`fullfile(pwd,'Lyrebird Data',...)`) and prefer vectorized calculations over loops for centroid math.

## Testing Guidelines
Validate each change against the bundled datasets: run static and dynamic scripts using `Lyrebird Data/` and `Robot traces/` inputs, and compare displacement plots against current MATLAB figures or saved text outputs (`Elekta/Persistent_output.txt`). Name any new regression checks `*_test.m` and store them beside the function under test. Record observed maximum deviations in comments to aid future comparisons.

## Commit & Pull Request Guidelines
Match the existing concise, imperative commit style seen in `git log` (`fixed`, `data acquired`). Provide descriptive bodies when touching multiple workflows, and reference issue IDs when available. Pull requests should summarize the QA scenario covered, list MATLAB commands used for verification, and attach representative plots or numeric deltas so reviewers can reproduce results quickly.
