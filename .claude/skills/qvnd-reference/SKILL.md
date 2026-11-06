---
name: qvnd-reference
description: Bibliography and source links for the Q-VND CVRP project — the papers the method and experimental protocol are modelled on, and where the benchmark instances and their known optima come from. Use when citing prior work, writing the paper, or tracing where a design choice or instance came from.
---

# References

- Jovanovic, Gavrilovic, Matijevic: Q-Learning-Based Adaptive Operator Selection
  in GA for 0/1 Knapsack, SYM-OP-IS 2026  [methodological template]
- Matijevic: GVNS for TD-EVRP-STW, IJIEC 2023  [reference for neighborhoods and GVNS design]
  https://www.growingscience.com/ijiec/Vol14/IJIEC_2023_8.pdf
- Hansen, Mladenovic, Todosijevic, Hanafi: VNS basics and variants, EURO J 2017
- Sutton, Barto: Reinforcement Learning, 2nd ed., chapter 6.5 (Q-learning)
- CVRPLIB: vrp.galgos.inf.puc-rio.br  [instances + known optima]

## Instance source

Augerat set A, all 27 instances, downloaded from the CVRPLIB listing at
https://galgos.inf.puc-rio.br/cvrplib/en/instances/1 (set A is `instance-set/4`;
individual files are `/cvrplib/en/download/instance/<id>`).

Each .vrp file carries its own known optimum in the COMMENT line, so no separate
table of optima is kept — parse_vrp reads it into inst["optimum"].
