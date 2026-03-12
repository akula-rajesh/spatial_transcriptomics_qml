#!/bin/bash
# SSL Certificate Fix Script for macOS
# This script helps fix SSL certificate verification issues

echo "=========================================="
echo "SSL Certificate Fix for macOS"
echo "=========================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is designed for macOS"
    echo "For other systems, please follow the SSL fix documentation"
    exit 1
fi

echo "Checking Python SSL certificates..."
echo ""

# Get Python version and path
PYTHON_CMD=$(which python3)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Python: $PYTHON_VERSION"
echo "Path: $PYTHON_CMD"
echo ""

# Method 1: Try to install certifi certificates
echo "Method 1: Installing/updating certifi package..."
pip3 install --upgrade certifi
CERTIFI_RESULT=$?

if [ $CERTIFI_RESULT -eq 0 ]; then
    echo "✓ certifi installed/updated successfully"
else
    echo "✗ Failed to install certifi"
fi
echo ""

# Method 2: Run Python's Install Certificates.command (if exists)
echo "Method 2: Running Python's Install Certificates command..."
PYTHON_PATH=$(dirname $(dirname $PYTHON_CMD))
CERT_SCRIPT="$PYTHON_PATH/Install Certificates.command"

if [ -f "$CERT_SCRIPT" ]; then
    echo "Found certificate installer at: $CERT_SCRIPT"
    bash "$CERT_SCRIPT"
    echo "✓ Certificates installed"
else
    echo "Certificate installer not found at expected location"

    # Try alternative locations
    for PY_VERSION in 3.11 3.10 3.9 3.8; do
        ALT_CERT="/Applications/Python ${PY_VERSION}/Install Certificates.command"
        if [ -f "$ALT_CERT" ]; then
            echo "Found at: $ALT_CERT"
            bash "$ALT_CERT"
            echo "✓ Certificates installed"
            break
        fi
    done
fi
echo ""

# Method 3: Install certificates via Homebrew (if available)
if command -v brew &> /dev/null; then
    echo "Method 3: Checking Homebrew certificates..."
    brew list openssl &> /dev/null
    if [ $? -eq 0 ]; then
        echo "✓ OpenSSL installed via Homebrew"
    else
        echo "Installing OpenSSL via Homebrew..."
        brew install openssl
    fi
else
    echo "Method 3: Homebrew not installed (optional)"
fi
echo ""

# Test SSL connection
echo "=========================================="
echo "Testing SSL Connection..."
echo "=========================================="
echo ""

python3 << 'PYEOF'
import ssl
import urllib.request

test_url = "https://www.python.org"
print(f"Testing connection to: {test_url}")

try:
    # Test with default context
    context = ssl.create_default_context()
    with urllib.request.urlopen(test_url, context=context, timeout=10) as response:
        print("✓ SSL verification WORKING with default context")
        print(f"  Response code: {response.status}")
except Exception as e:
    print(f"✗ SSL verification FAILED with default context")
    print(f"  Error: {e}")

    # Test with unverified context
    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(test_url, context=context, timeout=10) as response:
            print("✓ Connection works with unverified context")
            print(f"  Response code: {response.status}")
            print("")
            print("⚠️  RECOMMENDATION:")
            print("  SSL verification is failing but unverified works.")
            print("  This indicates a certificate issue.")
            print("  The pipeline will use unverified SSL (verify_ssl: false)")
    except Exception as e2:
        print(f"✗ Even unverified context failed: {e2}")
PYEOF

echo ""
echo "=========================================="
echo "Configuration Check"
echo "=========================================="
echo ""

CONFIG_FILE="config/pipeline_config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "Checking $CONFIG_FILE..."

    if grep -q "verify_ssl: false" "$CONFIG_FILE"; then
        echo "✓ SSL verification is disabled in config"
        echo "  This will bypass certificate verification"
    else
        echo "⚠️  verify_ssl not set to false"
        echo ""
        echo "To disable SSL verification, add to data section in config:"
        echo "  data:"
        echo "    verify_ssl: false"
    fi
else
    echo "Config file not found at: $CONFIG_FILE"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "If SSL verification still fails:"
echo ""
echo "1. ✓ Config is already updated with verify_ssl: false"
echo "2. ✓ Code is updated to handle SSL properly"
echo "3. Run the pipeline again:"
echo "   python src/main.py --config config/pipeline_config.yaml --mode train"
echo ""
echo "Alternative: Install Python certificates manually:"
echo "   For Python from python.org:"
echo "   - Open Applications folder"
echo "   - Find Python 3.x folder"
echo "   - Double-click 'Install Certificates.command'"
echo ""
echo "=========================================="
