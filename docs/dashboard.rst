Monitoring Dashboard
====================

This page describes the Metabase dashboard used to monitor the YOM
recommender system in production. It walks through each section of the
dashboard, explains what the charts show, and defines every metric that
appears on them.

The data shown in the dashboard is synthetic test data. Time ranges may
therefore be inconsistent between sections (e.g. October 2025 in some
panels, January 2024 in others), and day-over-day deltas for partial
days can look anomalous. This is expected and does not reflect a bug in
the pipeline.


Metric Definitions
------------------

All metrics below are computed globally across the entire system,
aggregating all recommendations, impressions, clicks, visits, and
purchases, unless stated otherwise. :math:`B` denotes a bundle (a set
of jointly recommended products).

**Click-Through Rate (CTR)** — share of displayed recommendations that
were clicked.

.. math::

   \text{CTR} = \frac{\#\,\text{Recommendation Clicks}}{\#\,\text{Displayed Recommendations}}

**Click-Through Purchase (CTP)** — share of recommendation clicks that
ultimately led to a purchase.

.. math::

   \text{CTP} = \frac{\#\,\text{Recommendation Purchases}}{\#\,\text{Recommendation Clicks}}

**Bundle Take Rate (BTR)** — share of displayed bundles that were
purchased.

.. math::

   \text{BTR} = \frac{\#\,\text{Bundles purchased}}{\#\,\text{Bundles displayed}}

**Conversion Rate (CR)** — share of bundle visits that ended in a
purchase.

.. math::

   \text{CR} = \frac{\#\,\text{Bundles purchased}}{\#\,\text{Bundle visits}}

**Absolute Additional Sales** — number of purchases attributed to the
recommender, i.e. purchases that followed a recommendation click. On
the dashboard this is aggregated in local currency (CLP) using the
bundle price.

.. math::

   \text{Absolute Additional Sales} = \#\,\text{Recommendation Purchases}

**Count of Purchases(B)** — number of orders that contain bundle
:math:`B`. With :math:`O` the set of all orders:

.. math::

   \text{Count of Purchases}(B) = \sum_{o \in O} \text{contains}(o, B)

**Revisits(B)** — number of clicks on bundle :math:`B`. With :math:`C`
the set of all clicks:

.. math::

   \text{Revisits}(B) = \bigl|\{\,c \in C \mid \text{isBundleClick}(c, B)\,\}\bigr|


Section 1 — Executive Overview
------------------------------

A row of top-level KPIs intended as the dashboard's at-a-glance health
check. Each tile shows the current value, the reference date, and the
delta versus the previous day.

- **Absolute Additional Sales** — revenue attributed to the recommender
  for the reporting day.
- **Bundle Take Rate** — BTR as defined above.
- **Click Through Rate** — CTR as defined above.
- **Click Through Purchase** — CTP as defined above.
- **Revisits** — total number of bundle clicks for the reporting day
  (sum of ``Revisits(B)`` over all bundles).

Partial-day snapshots can produce large negative deltas (e.g. a −99 %
drop in Revisits) simply because the day has not finished yet. This is
a known artefact of the live reporting window, not a regression.


Section 2 — Funnel Analysis
---------------------------

A single conversion funnel that tracks users through the four stages of
a recommendation:

1. **impression** — a recommendation was displayed.
2. **click** — the user clicked the recommendation.
3. **add_to_basket** — the recommended bundle was added to the basket.
4. **purchased** — the bundle was actually purchased.

The percentages shown under each stage are computed relative to the
number of impressions at the top of the funnel, so the final stage
value corresponds directly to the overall impression-to-purchase
conversion. The impression-to-click ratio at stage 2 equals the global
CTR.


Section 3 — Segmentation
------------------------

Breaks down engagement by commercial segment, product category, and
geography.

**Click Through Rate by Segment** — CTR over time, split by sales
channel (``Distribuidores``, ``Foodservice``, ``Mayorista``, ``Ruta``,
``Supermercados``). Used to spot channels where the recommender
under- or over-performs.

**Click Attribution by Category** — donut chart showing which product
categories receive the clicks. The total in the centre is the click
count over the reporting window. Useful for identifying category bias
in the recommendations.

**Click-Through-Rate by Region** — CTR per region (Chilean cities).
Highlights geographic performance differences that may warrant
region-specific re-ranking features.


Section 4 — Market Basket Analysis
----------------------------------

Diagnostics on basket composition, grouped by the same channel and
origin dimensions used during training. These charts are primarily for
the data science team to validate that basket statistics in production
match the distribution the model was trained on.

**Count of Products in the Basket** — box plot of basket size per
channel. Used to check whether basket-size distributions are stable
over time.

**Amplitude by Segment** — box plot of the number of distinct product
categories per basket, grouped by origin (``ZHH1``, ``ZCA2``, ``ZB2P``,
``ZPRE``, ``Ruta``, ``Mayorista``, ``Foodservice``). Measures how
diverse baskets are within each segment.

**Amplitude Over Time** — average category amplitude per day, split by
origin. Useful for detecting drift in basket diversity.


Section 5 — More Data
---------------------

Supplementary time-series views that complement the Executive Overview.

**Additional Sales over Time** — daily sum of attributed bundle
revenue. The day-level counterpart to the *Absolute Additional Sales*
KPI.

**Purchases** — daily count of recommendation-driven purchases.

**Engagement Metrics Over Time** — CTR, CTP, and CR plotted at hourly
resolution. This is the primary view for detecting short-term anomalies
in engagement; isolated spikes typically coincide with low-volume hours
where the denominator collapses.

**Count of Purchases by Productcategory** — daily purchase counts
broken down by product category. Used to verify that category-level
purchase volumes remain consistent with expectations.


Section 6 — Top 10 Bundles
--------------------------

Bar chart of the ten bundles with the highest purchase count over the
reporting window, ranked by ``Count of Purchases(B)``. Useful for:

- Identifying which bundles drive the majority of recommendation
  revenue.
- Sanity-checking that top bundles are commercially sensible
  combinations rather than artefacts of the data.
- Feeding back into catalogue and merchandising decisions.
