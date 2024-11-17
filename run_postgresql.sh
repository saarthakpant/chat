#!/bin/bash

# =============================================================================
# Script Name: run_postgresql.sh
# Description: Starts and enables PostgreSQL service and verifies connectivity.
# Author: OpenAI ChatGPT
# Date: 2024-04-27
# =============================================================================

# =============================================================================
# Configuration Variables
# =============================================================================

# PostgreSQL service name
SERVICE_NAME="postgresql"

# PostgreSQL connection parameters
DB_NAME="llm_icl_db"
DB_USER="llm_user"
DB_PASSWORD="your_secure_password"  # Replace with the same password used in setup_postgresql.sh
DB_HOST="localhost"
DB_PORT="5432"

# =============================================================================
# Function Definitions
# =============================================================================

# Function to start and enable PostgreSQL service
start_enable_service() {
    echo "-----------------------------------------------"
    echo "Starting PostgreSQL service..."
    sudo systemctl start $SERVICE_NAME
    if [ $? -ne 0 ]; then
        echo "❌ Failed to start PostgreSQL service."
        exit 1
    else
        echo "✅ PostgreSQL service started successfully."
    fi

    echo "Enabling PostgreSQL service to start on boot..."
    sudo systemctl enable $SERVICE_NAME
    if [ $? -ne 0 ]; then
        echo "❌ Failed to enable PostgreSQL service."
        exit 1
    else
        echo "✅ PostgreSQL service enabled to start on boot."
    fi
}

# Function to check PostgreSQL service status
check_service_status() {
    echo "-----------------------------------------------"
    echo "Checking PostgreSQL service status..."
    sudo systemctl status $SERVICE_NAME --no-pager
}

# Function to verify database connectivity
verify_connectivity() {
    echo "-----------------------------------------------"
    echo "Verifying database connectivity..."

    PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -d $DB_NAME -h $DB_HOST -p $DB_PORT -c "\dt" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Unable to connect to the PostgreSQL database."
        exit 1
    else
        echo "✅ Successfully connected to the PostgreSQL database."
    fi
}

# =============================================================================
# Main Script Execution
# =============================================================================

echo "==============================================="
echo "        PostgreSQL Run Script"
echo "==============================================="

# Confirm with the user before proceeding
read -p "⚠️ This script will start and enable the PostgreSQL service, then verify connectivity. Do you want to continue? (y/N): " confirmation
case "$confirmation" in
    [yY][eE][sS]|[yY]) 
        echo "Proceeding with PostgreSQL service management..."
        ;;
    *)
        echo "❌ Run script aborted by user."
        exit 1
        ;;
esac

# Start and enable PostgreSQL service
start_enable_service

# Check PostgreSQL service status
check_service_status

# Verify database connectivity
verify_connectivity

echo "==============================================="
echo "      PostgreSQL Service is Running and Verified"
echo "==============================================="
