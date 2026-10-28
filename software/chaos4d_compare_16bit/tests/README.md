# tests

This folder contains automated verification for the project.

Typical coverage:

- metric behavior
- parameter synthesis checks
- experiment pipeline smoke tests
- official-NIST-related helper validation

Smoke tests that need filesystem outputs should write temporary artifacts under
`temp/tmp_*`, not as new top-level scratch directories.

Add tests here whenever a new reusable module or experiment pipeline is added.
