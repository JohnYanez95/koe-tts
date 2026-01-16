#!/bin/bash
# Simple notification helper using ntfy.sh (free push notifications)
#
# Setup:
#   1. Install ntfy app on phone (Android/iOS)
#   2. Subscribe to your topic (e.g., "jyanez-tts-training")
#   3. That's it - no account needed
#
# Usage:
#   ./notify.sh "Training complete!"
#   ./notify.sh "ERROR: Training failed" "high"

TOPIC="${NTFY_TOPIC:-jyanez-tts}"  # Set your own topic name
MESSAGE="${1:-Notification from jp-tts}"
PRIORITY="${2:-default}"  # default, low, high, urgent

curl -s \
  -H "Priority: $PRIORITY" \
  -H "Tags: robot" \
  -d "$MESSAGE" \
  "https://ntfy.sh/$TOPIC" > /dev/null

echo "Notification sent to ntfy.sh/$TOPIC"
