WITH causal AS (
      SELECT *, LAG(speed,1) OVER w AS speed_lag1, LAG(global_time,1) OVER w AS time_lag1,
LAG(speed,3) OVER w AS speed_lag3, LAG(global_time,3) OVER w AS time_lag3,
LAG(speed,5) OVER w AS speed_lag5, LAG(global_time,5) OVER w AS time_lag5,
LAG(speed,10) OVER w AS speed_lag10, LAG(global_time,10) OVER w AS time_lag10,
        AVG(speed) OVER (PARTITION BY track ORDER BY global_time ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS past_mean,
        AVG(speed*speed) OVER (PARTITION BY track ORDER BY global_time ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS past_mean_square
      FROM frames WINDOW w AS (PARTITION BY track ORDER BY global_time)
    ) SELECT * FROM causal WHERE speed_lag10 IS NOT NULL ORDER BY location,track,global_time;