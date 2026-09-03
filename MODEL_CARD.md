# Model card — customer outreach propensity scorer

## Purpose

Rank eligible retail customers for a limited re-engagement queue. This is a portfolio prototype illustrating modelling-ready data, temporal controls, batch scoring and usable commercial output.

## Data and unit of analysis

Source: Chen, D. (2015), *Online Retail*, UCI Machine Learning Repository, https://doi.org/10.24432/C5BW33 (CC BY 4.0). The unit is a **customer at a weekly scoring snapshot**. Raw transactions are intentionally not committed to this repository.

## Target and decision time

At each score date, `target = 1` if the customer makes at least one positive purchase in the following 28 days. Features are computed solely from the prior 90 days. The final eight weekly score dates are held out, so the evaluation does not let the model train on future customer behaviour.

## Features

- purchases, spend, units and active days in the previous 90 days
- average order value and days since most recent purchase
- most recently observed country
- historical return units and return rate

## Model and validation

Logistic regression, with median imputation and standardisation for numeric inputs, and most-frequent imputation plus one-hot encoding for country. The chronological holdout begins on 20 September 2011.

## Observed holdout results

| Metric | Value |
| --- | ---: |
| Train rows | 54,765 |
| Test rows | 18,076 |
| ROC-AUC | 0.687 |
| Average precision | 0.621 |
| Brier score | 0.211 |
| Base conversion rate | 38.0% |
| Precision at top 10% | 81.4% |
| Lift at top 10% | 2.14× |

## Operational controls required before real use

Apply consent, opt-out and suppression filters before the model; resolve identities at person and account level; cap contact frequency; monitor missingness, score drift, conversion and complaints; retrain only after outcome maturity; and test incremental campaign impact with a randomised control group.

## Limits

The data represents retail transactions, not B2B contacts, calls or CRM opportunities. Scores estimate association with a future purchase, not causal campaign uplift. The GCP/CRM path described in the README is an implementation proposal, not a deployed claim.

