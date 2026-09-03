-- Proposed BigQuery feature-table pattern; not executed in this local project.
-- Every feature must be available before score_date.
CREATE OR REPLACE TABLE analytics.customer_outreach_features AS
SELECT
  c.customer_id,
  @score_date AS score_date,
  DATE_DIFF(@score_date, MAX(o.order_date), DAY) AS recency_days,
  COUNTIF(o.order_date >= DATE_SUB(@score_date, INTERVAL 90 DAY)) AS purchases_90d,
  SUM(IF(o.order_date >= DATE_SUB(@score_date, INTERVAL 90 DAY), o.net_revenue, 0)) AS spend_90d
FROM crm.customers c
LEFT JOIN sales.orders o
  ON c.customer_id = o.customer_id
 AND o.order_date < @score_date
GROUP BY 1, 2;

