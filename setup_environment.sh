echo "Setting up Python environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment (for macOS)
source venv/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install packages
pip install -r requirements.txt

echo "Installation complete!"
echo "Activate virtual environment with: source venv/bin/activate"