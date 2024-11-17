#!/bin/bash

# =============================================================================
# Script Name: setup_postgresql.sh
# Description: Installs and configures PostgreSQL with pgvector on Ubuntu 24.04.
# Author: OpenAI ChatGPT
# Date: 2024-04-27
# =============================================================================

# =============================================================================
# Configuration Variables
# =============================================================================

# PostgreSQL configuration
DB_NAME="llm_icl_db"
DB_USER="llm_user"
DB_PASSWORD="your_secure_password"  # Replace with a strong password

# pgvector configuration
PGVECTOR_VERSION="0.8.0"  # Replace with the latest version if available

# =============================================================================
# Function Definitions
# =============================================================================

# Function to install PostgreSQL and necessary dependencies
install_postgresql() {
    echo "-----------------------------------------------"
    echo "Updating package lists..."
    sudo apt update -y
    echo "-----------------------------------------------"

    echo "Installing PostgreSQL, contrib package, and development headers..."
    sudo apt install -y postgresql postgresql-contrib postgresql-server-dev-all git build-essential
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install PostgreSQL and dependencies."
        exit 1
    else
        echo "✅ PostgreSQL and dependencies installed successfully."
    fi
}

# Function to install pgvector
install_pgvector() {
    echo "-----------------------------------------------"
    echo "Installing pgvector extension..."

    # Navigate to a temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR" || { echo "❌ Failed to create temporary directory."; exit 1; }

    # Clone the pgvector repository
    git clone --branch v$PGVECTOR_VERSION https://github.com/pgvector/pgvector.git
    if [ $? -ne 0 ]; then
        echo "❌ Failed to clone pgvector repository."
        exit 1
    fi

    cd pgvector || { echo "❌ Failed to enter pgvector directory."; exit 1; }

    # Build and install pgvector
    make
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build pgvector."
        exit 1
    fi

    sudo make install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install pgvector."
        exit 1
    else
        echo "✅ pgvector installed successfully."
    fi

    # Clean up
    cd ~
    rm -rf "$TEMP_DIR"
}

# Function to configure PostgreSQL user and database
configure_postgresql() {
    echo "-----------------------------------------------"
    echo "Configuring PostgreSQL user and database..."

    sudo -i -u postgres psql <<EOF
DO \$\$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_user WHERE usename = '$DB_USER'
    ) THEN
        CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
    END IF;
END
\$\$;

DO \$\$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_database WHERE datname = '$DB_NAME'
    ) THEN
        CREATE DATABASE $DB_NAME OWNER $DB_USER;
    END IF;
END
\$\$;

GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOF

    if [ $? -ne 0 ]; then
        echo "❌ Failed to configure PostgreSQL user and database."
        exit 1
    else
        echo "✅ PostgreSQL user and database configured successfully."
    fi
}

# Function to enable pgvector extension in the database
enable_pgvector_extension() {
    echo "-----------------------------------------------"
    echo "Enabling pgvector extension in the database..."

    sudo -i -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to enable pgvector extension."
        exit 1
    else
        echo "✅ pgvector extension enabled successfully."
    fi
}

# =============================================================================
# Main Script Execution
# =============================================================================

echo "==============================================="
echo "        PostgreSQL and pgvector Setup Script"
echo "==============================================="

# Confirm with the user before proceeding
read -p "⚠️ This script will install PostgreSQL, pgvector extension, create a database and user. Do you want to continue? (y/N): " confirmation
case "$confirmation" in
    [yY][eE][sS]|[yY]) 
        echo "Proceeding with setup..."
        ;;
    *)
        echo "❌ Setup aborted by user."
        exit 1
        ;;
esac

# Install PostgreSQL and dependencies
#install_postgresql

# Install pgvector extension
install_pgvector

# Configure PostgreSQL user and database
configure_postgresql

# Enable pgvector extension in the database
enable_pgvector_extension

echo "==============================================="
echo "      PostgreSQL and pgvector Setup Completed Successfully"
echo "==============================================="
