#!/bin/bash
# Wrapper to run any command and notify on completion/failure
#
# Usage:
#   ./run_with_notify.sh python train.py --epochs 100
#   ./run_with_notify.sh bash scripts/download_jsut.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPIC="${NTFY_TOPIC:-jyanez-tts}"

echo "Running: $@"
echo "Notifications will be sent to: ntfy.sh/$TOPIC"
echo "---"

# Run the command
"$@"
EXIT_CODE=$?

# Notify based on result
if [ $EXIT_CODE -eq 0 ]; then
    curl -s -H "Priority: default" -H "Tags: white_check_mark" \
        -d "✅ Command succeeded: $1" \
        "https://ntfy.sh/$TOPIC" > /dev/null
    echo "---"
    echo "Command completed successfully. Notification sent."
else
    curl -s -H "Priority: high" -H "Tags: x" \
        -d "❌ Command FAILED (exit $EXIT_CODE): $1" \
        "https://ntfy.sh/$TOPIC" > /dev/null
    echo "---"
    echo "Command failed with exit code $EXIT_CODE. Notification sent."
fi

exit $EXIT_CODE
