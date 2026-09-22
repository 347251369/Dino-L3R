# Final model status

The active pipeline contains two stages: **GSP-DINO** and **OCRE**. Versioned
experiment labels are not used in configurations, reports, or artifact names.

## FLARE22

- Split: 45 train / 5 fixed development-test cases
- Test IDs: 0029, 0013, 0042, 0010, 0034
- Split SHA256: `192ef2ece6903ba67e2f1a8ba440a3abf42982d57b3c0526a24b79638a0f7044`
- GSP checkpoint SHA256: `27e74a73a6f13b0dd1caf2e826aaeade07e2154cb5c0810ea45338d074bf3445`
- GSP overall Dice: 0.924432
- OCRE checkpoint SHA256: `e8d58eb01eecae721abe7d190eaf3c07fff877cf126d6406845fcea947a94ad1`
- OCRE mean Dice: 0.940294
- OCRE ASSD / HD95: 0.586720 mm / 2.711605 mm

The fixed FLARE22 test cases were viewed during development. These values are
development-set evidence, not an independent blind-test estimate.

## BTCV

- Split: 24 train / 6 fixed cases
- Test IDs: 0006, 0028, 0031, 0032, 0033, 0036
- GSP checkpoint SHA256: `71e4db77579ae968b8f3fbb65eca94079dcb6d2042682e117556ccdfe28c4fa3`
- OCRE checkpoint SHA256: `5f646d2f8dd0c532f714b91dd6f446619c9582d040d690ab6b4f6ebf71d8ccae`
- OCRE mean Dice: 0.838709
- OCRE ASSD / HD95: 1.813237 mm / 9.378460 mm

The final OCRE implementation was trained for 24 epochs and evaluated on all
six fixed cases. GSP-DINO was reused read-only throughout the run.
