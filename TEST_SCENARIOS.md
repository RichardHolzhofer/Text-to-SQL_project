# Factual Correctness Test Scenarios

This document outlines 20 test scenarios designed to validate the Text-to-SQL application's accuracy, routing, and adherence to system limits.

## Test Suite Overview

The suite is divided into four key categories to ensure all paths of the graph are exercised.

| Category | Description | Key Limits Tested |
| :--- | :--- | :--- |
| **Standard NL** | Natural Language summaries of structured data. | `standard_nl_limit` (100) |
| **Standard Tab** | Tabular responses for data retrieval requests. | `standard_tab_limit` (10) |
| **Semantic** | Vector-search based queries for review sentiment/content. | `semantic_nl_limit` (100), `semantic_tab_limit` (10) |
| **Boundaries** | Stress tests for truncation and safety mechanisms. | `safety_limit` (5000), All Truncation Limits |

---

## Detailed Scenarios

### 1. Standard Query - Natural Language (NL)
*Focus: Accuracy of summarization and basic SQL logic.*

- **NL-01: Geographic Distribution**  
  *Question*: "Summarize the customer distribution across different Brazilian states."  
  *Goal*: Validate state-level grouping and NL summarization.

- **NL-02: Customer Loyalty Analysis**  
  *Question*: "Explain the difference in average order value between new and returning customers."  
  *Goal*: Test logic involving `is_returning_customer` and cross-category explanation.

- **NL-03: Revenue Trends**  
  *Question*: "Provide a summary of revenue trends for the first quarter of 2018."  
  *Goal*: Test date filtering and temporal summarization.

- **NL-04: Regional Performance Comparison**  
  *Question*: "Compare the delivery performance and delay metrics between 'SP' and 'RJ' states."  
  *Goal*: Test comparative analysis across multiple metrics.

- **NL-05: Category Performance**  
  *Question*: "Summarize the top-performing product categories in terms of total sales volume."  
  *Goal*: Test ranking and categorical summarization.

### 2. Standard Query - Tabular (Tab)
*Focus: Data retrieval accuracy and 10-row tabular formatting.*

- **TAB-01: City-Level Customer Count**  
  *Question*: "Give me a table of the top 5 cities with the most customers."  
  *Goal*: Validate tabular output and row-count accuracy.

- **TAB-02: Order Status Breakdown**  
  *Question*: "Show a table with the count of orders for each current order status."  
  *Goal*: Test categorical grouping in a tabular format.

- **TAB-03: High-Value Customers**  
  *Question*: "List the top 5 customers with the highest lifetime value (LTV) in a table."  
  *Goal*: Test multi-table join and sorting in tables.

- **TAB-04: Freight Cost Table**  
  *Question*: "Create a table showing the average freight value paid for each state."  
  *Goal*: Test geographic metric retrieval in tables.

- **TAB-05: Payment Method Analysis**  
  *Question*: "Provide a table showing the total amount paid via each payment type."  
  *Goal*: Test financial data extraction into a table.

### 3. Semantic Search (Review Search)
*Focus: Vector search routing and review content synthesis.*

- **SEM-01: Shipping Complaints Summary**  
  *Question*: "What are the most common complaints customers have about shipping and delivery?"  
  *Goal*: Test semantic intent routing and thematic summary.

- **SEM-02: Positive Category Sentiment**  
  *Question*: "Summarize the positive feedback specifically for 'electronics' products."  
  *Goal*: Test filtered semantic search (category + sentiment).

- **SEM-03: Damaged Items Review Table**  
  *Question*: "Give me a table of 5 reviews that specifically mention 'damaged' or 'broken' items."  
  *Goal*: Test semantic search with tabular output format.

- **SEM-04: Product Quality Synthesis**  
  *Question*: "What is the general feedback regarding the quality of 'health_beauty' products?"  
  *Goal*: Test broad thematic synthesis for a specific category.

- **SEM-05: Service Issue Identification**  
  *Question*: "Find and summarize reviews that mention issues with 'customer service' or 'refunds'."  
  *Goal*: Test semantic search for specific service-related keywords.

### 4. System Boundaries & Safety Limits
*Focus: Truncation logic and the 5000-record safety cap.*

- **BND-01: Safety Limit Trigger**  
  *Question*: "List every single order ID and its status in the database."  
  *Goal*: **Trigger Safety Limit (>5000)**. Verify the disclaimer and SQL provision.

- **BND-02: NL Truncation Test**  
  *Question*: "Summarize all orders placed in the year 2017."  
  *Goal*: **Trigger NL Truncation (>100)**. Verify summary of first 100 and total count mention.

- **BND-03: Tabular Truncation Test**  
  *Question*: "Show a table of the first 50 customers sorted by registration date."  
  *Goal*: **Trigger Tab Truncation (>10)**. Verify display of only 10 rows and download disclaimer.

- **BND-04: Semantic NL Truncation**  
  *Question*: "Summarize all reviews that mention the word 'good'."  
  *Goal*: **Trigger Semantic NL Truncation (>100)**. Verify limit compliance for vector results.

- **BND-05: Semantic Tab Truncation**  
  *Question*: "Give me a table of all reviews that mention 'delivery'."  
  *Goal*: **Trigger Semantic Tab Truncation (>10)**. Verify limit compliance for vector tabular output.
