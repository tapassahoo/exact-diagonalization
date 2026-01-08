#!/bin/bash

set -e

# -------- Paths --------
LOCAL_DIR="/Users/tapas/academic-project/exact-diagonalization/pkg_monomer_rotor/"
REMOTE="tapas:/home/tapas/academic-project/exact-diagonalization/pkg_monomer_rotor/"
DRY_RUN=False   # set to true for preview
# -----------------------

# Sanity check
[ -d "$LOCAL_DIR" ] || { echo "Local directory not found."; exit 1; }

# Record Git version (if applicable)
if [ -d "$LOCAL_DIR/.git" ]; then
    git -C "$LOCAL_DIR" rev-parse HEAD > "$LOCAL_DIR/deploy_commit.txt"
fi

echo "--------------------------------------------------"
echo "Deploying pkg_monomer_rotor"
echo "Local : $LOCAL_DIR"
echo "Remote: $REMOTE"
echo "Dry run: $DRY_RUN"
echo "--------------------------------------------------"

RSYNC_OPTS="-avz --delete --progress --itemize-changes"

[ "$DRY_RUN" = true ] && RSYNC_OPTS="$RSYNC_OPTS --dry-run"

rsync $RSYNC_OPTS \
      --exclude=".git" \
      --exclude=".gitignore" \
      --exclude="__pycache__/" \
      --exclude="*.pyc" \
      --exclude="*.log" \
      --exclude="output/" \
      "$LOCAL_DIR" \
      "$REMOTE"

echo "--------------------------------------------------"
echo "Deployment complete."
ssh tapas "cat /home/tapas/academic-project/exact-diagonalization/pkg_monomer_rotor/deploy_commit.txt 2>/dev/null || true"
echo "--------------------------------------------------"

