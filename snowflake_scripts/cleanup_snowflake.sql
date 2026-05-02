-- Snowflake Cleanup Script
-- Run this as ACCOUNTADMIN to delete all project resources

-- 1. Drop Users
DROP USER IF EXISTS ECOMMERCE_DBT_USER;
DROP USER IF EXISTS ECOMMERCE_AI_USER;

-- 2. Drop Roles
DROP ROLE IF EXISTS ECOMMERCE_TRANSFORMER;
DROP ROLE IF EXISTS ECOMMERCE_AGENT;

-- 3. Drop Warehouse
DROP WAREHOUSE IF EXISTS ECOMMERCE_WH;

-- 4. Drop Database (This also drops all schemas inside it)
DROP DATABASE IF EXISTS ECOMMERCE_ANALYTICS;

-- 5. Drop redundant schemas if they were created outside the database (unlikely but good for safety)
DROP SCHEMA IF EXISTS RAW_DATA;
DROP SCHEMA IF EXISTS STAGING;
DROP SCHEMA IF EXISTS INTERMEDIATE;
DROP SCHEMA IF EXISTS MART;

SELECT 'Snowflake cleanup complete! You are ready to run setup_snowflake.sql again.' AS result;
