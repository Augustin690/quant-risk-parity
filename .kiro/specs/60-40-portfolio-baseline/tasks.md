# Implementation Plan

- [x] 1. Implement baseline data fetching functionality
  - Create `fetch_baseline_data()` function in `data.py` that fetches SPY and IEF price data
  - Add baseline configuration constants for tickers and weights
  - Implement error handling and data validation for baseline tickers
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 2. Create baseline portfolio construction logic
  - Implement `run_baseline()` function in `backtest.py` that creates 60/40 static allocation
  - Add rebalancing logic that maintains 60% SPY, 40% IEF weights on rebalance dates
  - Calculate baseline portfolio returns using static weights and asset returns
  - Return weights DataFrame and portfolio returns Series consistent with main strategy format
  - _Requirements: 1.1, 1.2, 1.4_

- [ ]* 2.1 Write unit tests for baseline portfolio construction
  - Test static weight allocation accuracy (60/40 split)
  - Test rebalancing logic maintains target weights
  - Test return calculation matches expected portfolio performance
  - _Requirements: 1.1, 1.2, 1.4_

- [x] 3. Enhance performance metrics to include baseline comparison
  - Create `summarize_with_baseline()` function in `metrics.py` that compares strategy vs baseline
  - Calculate comparative metrics (Sharpe ratio difference, excess returns, etc.)
  - Format output DataFrame to show strategy and baseline side-by-side
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ]* 3.1 Write unit tests for baseline performance metrics
  - Test comparative metrics calculation accuracy
  - Test edge cases with missing or insufficient baseline data
  - Verify output format matches expected DataFrame structure
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 4. Update visualization to include baseline comparison
  - Modify `plot_equity_curve()` function to add baseline cumulative return line
  - Update baseline styling (gray dashed line, "60/40 Baseline" label)
  - Ensure baseline line uses same date range and scaling as strategy line
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 5. Enhance dashboard with baseline integration
  - Update `create_dashboard()` function to include baseline in equity curve plot
  - Modify legend and styling to accommodate baseline comparison
  - Ensure baseline integration doesn't break existing dashboard functionality
  - _Requirements: 3.1, 3.2, 3.3_

- [ ]* 5.1 Write integration tests for baseline visualization
  - Test dashboard generation with baseline data
  - Test plot styling and legend accuracy
  - Test error handling when baseline data is unavailable
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 6. Update CLI to support baseline functionality
  - Modify `run` command to fetch and calculate baseline alongside main strategy
  - Update `report` command to include baseline in performance summary and visualizations
  - Add baseline results to output files (baseline_returns.csv, baseline_summary.csv)
  - _Requirements: 1.1, 4.1, 4.2_

- [x] 7. Integrate baseline workflow end-to-end
  - Connect baseline data fetching, portfolio construction, and performance calculation
  - Ensure baseline calculation runs automatically when main strategy is executed
  - Verify baseline results are saved to outputs directory alongside strategy results
  - Test complete workflow from CLI command to final dashboard with baseline comparison
  - _Requirements: 1.1, 2.1, 3.1, 4.1_

- [ ]* 7.1 Write end-to-end integration tests
  - Test complete baseline workflow from data fetch to visualization
  - Test CLI commands with baseline functionality
  - Test output file generation and format
  - _Requirements: 1.1, 2.1, 3.1, 4.1_