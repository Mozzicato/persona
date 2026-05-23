# Human Evaluation Rubric

The rubric below is the one we use to score generated reviews and
recommendations. Scoring is on a 1-5 scale per criterion, then averaged.

## Part A — Generated Review (Task A)

| # | Criterion | 1 (poor) | 3 (acceptable) | 5 (excellent) |
|---|---|---|---|---|
| A1 | **Realism** — could a real Nigerian consumer have written this? | Robotic, marketing-copy tone | Mostly natural, occasional stiltedness | Indistinguishable from a real review |
| A2 | **Persona consistency** — does the voice match the user's history? | Wrong voice (e.g. analytical user suddenly writing pidgin) | Mostly matches with one off note | Voice, length, harshness, and slang level all align with persona |
| A3 | **Contextual awareness** — does it reflect the situation passed in? | Ignores context (rainy/traffic/festive flags) | References context briefly | Context is woven into the reaction naturally |
| A4 | **Rating-text coherence** — does the predicted rating match the tone of the text? | Mismatch (e.g. glowing text, 2-star rating) | Mostly aligned with one inconsistency | Tone and rating are tightly coupled |
| A5 | **Nigerian authenticity** — when pidgin/slang appears, is it correct? | Forced or wrong (e.g. "abeg" used incorrectly) | Adequate; one or two stiff phrases | Natural, idiomatic, scaled to the user's emotional intensity |
| A6 | **Emotional coherence** — does the stated emotional state match the text? | Contradicts the text | Mostly consistent | Text reads exactly as the stated emotional state would write |
| A7 | **Reasoning traceability** — does the `reasoning` field cite specific persona traits / memory tags / context flags? | Generic, ungrounded | Some citations | Every claim grounded in a specific source signal |

**Overall score**: average of A1-A7, weighted equally.

**Reject if**: any single criterion scores 1.

## Part B — Recommendations (Task B)

| # | Criterion | 1 (poor) | 3 (acceptable) | 5 (excellent) |
|---|---|---|---|---|
| B1 | **Relevance** — would the user plausibly want this item now? | No connection to persona or context | Reasonable but not personalised | Tightly fits both persona and current context |
| B2 | **Diversity** — does the top-N avoid repeating the same item idea? | All 5 recs are the same product idea | 3 distinct, 2 duplicates | All 5 are meaningfully different |
| B3 | **Explanation quality** — does the `reason` field cite the right signals? | Generic ("you might like this") | One or two grounded signals | Cites persona trait + memory state + retrieval source |
| B4 | **Trade-off honesty** — when the user has open friction (e.g. delivery), does the explanation acknowledge it? | Ignores friction | Mentions it once | Frames the recommendation as resolving or accepting the friction |
| B5 | **Re-rank vs retrieval** — does the final order make more sense than retrieval alone? | Top-3 looks identical to TF-IDF baseline | Some defensible re-ordering | Clearly better; bad items demoted, good unusual items lifted |
| B6 | **Context honoring** — does the recommendation reflect the auto-detected flags (salary_week, festive, …)? | Ignores them | Honors one | Honors all relevant flags |

**Overall score**: average of B1-B6, weighted equally.

## Scoring protocol

1. Two evaluators score independently.
2. Disagreements > 1 point on any criterion are discussed until within 1.
3. Final score per output = mean of the two evaluators.
4. Report **median** + **interquartile range** across 30 generated outputs
   (15 reviews + 15 recommendation sets), sampled from 5 distinct users
   × 3 distinct contexts.

## Calibration examples

### Realism A1 = 5
> Omo this shawarma sweet die. The chicken filling fresh and the bread no soggy at all. For the price, I no fit complain. I go order again next week sha.

### Realism A1 = 3
> The food was good and the service was nice. I would order again. The delivery was on time. Overall a positive experience.

### Realism A1 = 1
> ★★★★ Highly recommended! Five stars for excellent customer service and superior quality. Order today!

(The third reads like marketing copy, not a real reviewer.)
