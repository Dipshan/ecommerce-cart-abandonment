import importlib


def check_package(package_name, import_name=None):
    """Check if package is installed and get version"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown version')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: Not installed")
        return False


def main():
    print("Checking environment setup...")
    print("=" * 50)

    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("jupyter", "jupyter"),
        ("ipykernel", "ipykernel"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("dask", "dask"),
    ]

    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False

    print("=" * 50)
    if all_installed:
        print("✓ All packages installed successfully!")
    else:
        print("✗ Some packages are missing. Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
