A/B Test
========

This page describes the planned online A/B test that compares two
versions of the bundle recommender. The test is designed to determine,
with statistical significance, whether the new algorithm improves the
**Bundle Take Rate (BTR)** over the current production baseline.

The test has not been executed yet. All parameters below are sensible
defaults chosen to fit the volume of traffic on the platform
(approximately 1,000,000 orders per year). They should be revisited
once preliminary data from a pilot period is available.


Overview
--------

Two algorithms are compared:

- **Algorithm A** — the current production system (control / baseline).
- **Algorithm B** — the new candidate system (treatment).

The hypotheses to be tested are:

.. math::

   H_0\!: \mu_A = \mu_B \qquad H_1\!: \mu_B > \mu_A

where :math:`\mu` denotes the mean BTR of a group. The test is therefore
**one-sided**: the goal is to confirm that B is *better* than A, not
merely *different* from A. If the data does not support :math:`H_1`,
the production system stays on Algorithm A.


Primary Metric: Bundle Take Rate (BTR)
--------------------------------------

BTR is the share of displayed bundles that are actually purchased:

.. math::

   \text{BTR} = \frac{\#\,\text{Bundles purchased}}{\#\,\text{Bundles displayed}}

BTR is the right primary metric for this comparison because it
directly measures the bottom-line outcome the recommender is built to
drive: how often a displayed bundle ends up being bought. Unlike CTR
or CTP, which only capture one stage of the funnel, BTR collapses the
entire impression-to-purchase journey into a single number and
therefore aligns naturally with the *Absolute Additional Sales* KPI
that the business cares about.


Test Design
-----------

**Randomisation unit — Store**

Randomisation happens at the store level, not at the session or user
level. This matches the level at which the primary metric is reported
and the level at which most of the variance in purchasing behaviour
sits. It also avoids contamination effects: a single store always sees
the same algorithm, so neither operators nor end customers experience
inconsistent recommendations within a session.

**Stratified assignment**

Because store-level revenue is highly skewed (a small number of large
stores account for a large share of orders), purely random assignment
can produce groups with very different revenue profiles. Stores are
therefore stratified by historical revenue (e.g. quartiles) before
being randomly allocated within each stratum. This keeps the two
groups balanced on the dimension that matters most for the outcome.

**Allocation**

A 1:1 allocation between A and B is used. Equal allocation maximises
statistical power for a fixed total sample size and is the default
recommendation for a confirmatory comparison.

**Test parameters**

The following defaults apply. They can be tightened or relaxed
depending on how risk-averse the rollout decision should be.

- Significance level :math:`\alpha = 0.05`
- Power :math:`1 - \beta = 0.80`
- Minimum detectable effect (MDE): **10 % relative increase in BTR**
- Allocation ratio: 1:1
- Planned duration: approximately **7 days**

The 7-day window is a target rather than a hard constraint. The actual
duration is determined by the sample size calculation below and should
be confirmed from a short pilot run before the test starts.


Sample Size
-----------

The test is a cluster-randomised experiment: stores are clusters, and
bundle displays within a store are the units of observation. Standard
sample size formulas have to be inflated to account for the fact that
displays within the same store are not independent.

**Design Effect**

.. math::

   \text{DE} = 1 + (n - 1)\,\rho

where :math:`n` is the average number of bundle displays per store
during the test window and :math:`\rho` is the **intraclass correlation
(ICC)** — the share of variance in the take-or-not outcome that is
explained by the store the display happened in. As a rule of thumb,
display-level outcomes in e-commerce settings have an ICC in the range
of 0.01–0.05.

**Required bundle displays per group**

.. math::

   m = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot 2\sigma^2}{\Delta^2}
       \cdot \bigl(1 + (n - 1)\rho\bigr)

with:

- :math:`Z_{1-\alpha/2} = 1.96` (for :math:`\alpha = 0.05`)
- :math:`Z_{1-\beta} = 0.84` (for power = 0.80)
- :math:`\sigma^2 = p_A (1 - p_A)`, the variance of the binary
  take-or-not outcome for the baseline group
- :math:`\Delta = 0.10 \cdot p_A`, the absolute change implied by an
  MDE of 10 % relative

The baseline BTR :math:`p_A`, the ICC :math:`\rho`, and the average
displays per store :math:`n` are estimated from the most recent two
weeks of production data before the test starts. Plugging these into
the formula gives the required number of bundle displays per group,
which is then translated into a test duration based on the historical
display rate.

If the resulting duration exceeds two weeks, the MDE should be
increased (e.g. to 15 % relative). Detecting effects much smaller than
10 % relative is rarely justified by the business cost of the rollout
decision and inflates the sample size dramatically.


Guardrail Metrics and Stop Conditions
-------------------------------------

Even when BTR improves, the new algorithm can have negative
side-effects on the business as a whole — for example by reorganising
basket composition in a way that reduces overall revenue. Guardrail
metrics are monitored continuously during the test and trigger an
automatic stop if they breach a pre-defined threshold.

The thresholds below are **example values** and must be confirmed with
business stakeholders before the test goes live.

- **Total revenue (treatment vs. control)** — stop if the treatment
  group's daily revenue falls more than **5 %** below the control
  group's for two consecutive days.
- **Average Order Value (treatment vs. control)** — stop if AOV in the
  treatment group drops by more than **10 %** versus the baseline
  observed in the pre-test window.
- **Recommender error rate** — stop if more than **1 %** of
  recommendation requests fail or return empty results in either group.

Stop conditions are evaluated daily on the previous day's data. A
breach triggers an immediate rollback of the treatment group to
Algorithm A; the test is then closed and the breach is investigated
before any further experimentation.


Evaluation
----------

After the test window closes, BTR is computed per store, and the two
groups are compared with **Welch's t-test** on the store-level rates.
Welch's t-test is preferred over Student's t-test because the two
groups can have different variances, especially after stratification.

The standard error of the difference in means is

.. math::

   s_{\bar{X}} = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}

with degrees of freedom approximated by Welch-Satterthwaite

.. math::

   n_{df} \approx \frac{\left(\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}\right)^2}
                       {\dfrac{(s_1^2/n_1)^2}{n_1 - 1} + \dfrac{(s_2^2/n_2)^2}{n_2 - 1}}

and the test statistic

.. math::

   t = \frac{\bar{x}_B - \bar{x}_A}{s_{\bar{X}}}.

**Decision rule**

Reject :math:`H_0` in favour of :math:`H_1` if :math:`t > t_{\text{crit}}`,
where :math:`t_{\text{crit}}` is the one-sided critical value at
:math:`\alpha = 0.05` for :math:`n_{df}` degrees of freedom (≈ 1.645
for large :math:`n_{df}`). A rejection is interpreted as evidence that
Algorithm B improves BTR over Algorithm A and justifies the rollout.
A non-rejection means production stays on Algorithm A.


Logging Requirements
--------------------

Valid evaluation depends entirely on the logs. Every interaction with
the recommender during the test window must be recorded with enough
context to reconstruct the funnel and attribute outcomes to the
correct group.

Each recommendation event is logged with:

- ``store_id`` — the cluster the event belongs to.
- ``group`` — ``A`` or ``B``, the algorithm assignment of the store.
- ``algorithm_version`` — exact version string of the model serving
  the request, so the test can be re-run or audited later.
- ``event_type`` — one of ``impression``, ``click``,
  ``add_to_basket``, ``purchased``.
- ``bundle_id`` — the recommended bundle.
- ``session_id`` / ``click_id`` — to link clicks to the purchases they
  caused.
- ``timestamp`` — UTC, millisecond precision.

Logs are written in JSON for the event stream and aggregated nightly
into Parquet files for analysis. Store-to-group assignments are
recorded in a separate, immutable table at the start of the test and
are never modified during the run.


Monitoring Dashboard
--------------------

A dedicated Metabase dashboard tracks the A/B test in near-real-time.
It is the primary surface for spotting guardrail breaches and getting
an intuitive sense of how the test is progressing before the formal
evaluation at the end of the window. The dashboard uses the labels
``baseline_algo`` and ``advanced_algo`` for groups A and B
respectively.

**Section 1 — Executive Overview**

A row of side-by-side comparisons between the two groups.

- *CTR — A/B comparison over time* — hourly CTR for each algorithm.
  Sustained, parallel lines indicate that the experiment is running
  cleanly. Sudden divergence or convergence outside the expected
  effect size warrants a closer look at the logging pipeline before
  trusting downstream metrics.
- *Bundle Take Rate — by Algo* — daily BTR per group as a grouped bar
  chart. This is the primary visual readout of the test outcome and
  the chart to watch over the test window.
- *Absolute Additional Sales — Baseline Algo* and *— Advanced Algo* —
  two KPI tiles with daily attributed revenue per group, including
  day-over-day deltas. Used together with the revenue guardrail.

**Section 2 — Sankey Analysis**

A Sankey diagram of the conversion funnel split by algorithm. Each
group's flow from ``impression`` through ``click``, ``add_to_basket``
and ``purchased`` is shown side by side, making it visually obvious
where one algorithm is leaking users compared to the other and at
which stage the gap (if any) opens up. This view complements the BTR
chart: BTR shows *whether* the new algorithm is better, the Sankey
shows *where* the difference is coming from.
