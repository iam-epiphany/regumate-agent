# Rigorous v3 100-case learning summary

## Result

- Immutable first blind run: 80/100.
- Failed-case regression: 12/20 recovered, stored separately.
- Combined regression checkpoint: 92/100.
- Required-source evidence hit rate: 93.68%.
- CUDA runtime: NVIDIA GeForce RTX 5070 Laptop GPU, selected device `cuda`.
- Official 300 was not rerun at this checkpoint, by the revised acceptance order.

## What the batch exposed

1. Query planning confused workbook names, sheets, hierarchical row labels,
   columns and periods, turning valid cell locators into contradictory hard
   filters.
2. Quoted regulatory text containing words such as “决定” or “会计估计” could
   be misclassified as a request for subjective advice or prediction.
3. Multi-document requests were not always decomposed into source-isolated
   aspects, so one source could satisfy both planned branches.
4. Calculation operands inherited global period/column filters and could lose
   one side of a ratio or difference.  Display rounding could also leak into
   arithmetic.
5. When model generation was unavailable, extractive fallback could select a
   nearby clause or omit conditions, periods and exceptions even when the
   correct source was retrieved.

## General production improvements

- The query planner now keeps source title, sheet, row, column and operand
  selectors distinct; explicit cell locators override inferred period filters.
- Explicitly quoted multi-document requests produce one bounded aspect per
  source and quoted anchor.
- Boundary classification removes document titles and quoted source text before
  detecting advice, decisions and forecasts.
- Structured spreadsheet calculations use Decimal-compatible source values,
  preserve operand-specific selectors, validate dimensions and apply display
  rounding only after calculation.
- Definition queries add definition-language retrieval anchors, and extractive
  fallback ranks definition clauses and preserves a larger evidence excerpt.
- Citation and grounding validation remain mandatory for both generated and
  degraded answers.

## Measured effect

- Table lookup improved from 10/15 to 15/15.
- Rule/scope improved from 10/12 to 12/12.
- Single-document synthesis improved from 6/8 to 8/8.
- Table calculation improved from 8/10 to 9/10.
- Policy/reporting joint judgement improved from 4/5 to 5/5.
- Required-source evidence hit rate reached 93.68%.

## Residual risk

Cross-document evidence remains the weakest capability at 5/10.  Some failures
are genuine source-isolation or excerpt-selection misses; at least one frozen
case is itself non-unique because its quoted prefix occurs twice with different
continuations.  The second independent 100-case batch must use unique evidence
anchors and will be the basis for any further general repair.
