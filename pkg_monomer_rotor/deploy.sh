#!/bin/bash

set -e

# Paths
LOCAL_DIR="/Users/tapas/academic-project/exact-diagonalization/pkg_monomer_rotor/"
REMOTE_ALIAS="tapas"
REMOTE_DIR="/home/tapas/academic-project/exact-diagonalization/pkg_monomer_rotor/"

# Sanity check
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: Local directory does not exist."
    exit 1
fi

echo "--------------------------------------------------"
echo "Deploying pkg_monomer_rotor"
echo "Local : $LOCAL_DIR"
echo "Remote: $REMOTE_ALIAS:$REMOTE_DIR"
echo "--------------------------------------------------"

# Store local Git commit hash (if Git exists)
if [ -d "$LOCAL_DIR/.git" ]; then
    git -C "$LOCAL_DIR" rev-parse HEAD > "$LOCAL_DIR/deploy_commit.txt"
fi

# Rsync with clear reporting of synced files
rsync -avz --delete --progress --itemize-changes \
      --exclude=".git" \
      --exclude=".gitignore" \
      --exclude="__pycache__/" \
      --exclude="*.pyc" \
      --exclude="*.log" \
      --exclude="output/" \
      "$LOCAL_DIR" \
      "$REMOTE_ALIAS:$REMOTE_DIR"

echo "--------------------------------------------------"
echo "Deployment completed successfully."
echo "Remote version (if applicable):"
ssh "$REMOTE_ALIAS" "cat $REMOTE_DIR/deploy_commit.txt 2>/dev/null || echo 'No commit info found'"
echo "--------------------------------------------------"

