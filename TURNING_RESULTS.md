# Turning scenario quick check

This repository now includes a paper-inspired turning scenario named `turn_pi_over_4`.

## Reference setting

- desired forward speed: 0.35 m/s
- desired yaw target: pi/4 rad
- commanded yaw rate during the turn: 0.45 rad/s

## Observed closed-loop result

A quick closed-loop run of the Python baseline gave the following final values:

- final yaw: 0.78588 rad
- final yaw reference: 0.78540 rad
- final yaw error: 4.81e-4 rad
- final vx: 0.24748 m/s
- final vx reference: 0.24749 m/s
- final vy: 0.24749 m/s
- final vy reference: 0.24749 m/s
- RMS yaw tracking error: 1.57e-3 rad
- RMS vx tracking error: 7.74e-3 m/s

## Interpretation

The turning case is not a full reproduction of the MATLAB quadruped stack. However, the present Python baseline is able to:

- rotate the heading toward pi/4,
- settle close to the desired final yaw,
- follow the curved reference path,
- and maintain the commanded forward speed trend during the turn.
