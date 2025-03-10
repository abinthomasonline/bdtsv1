import os
import sys
from tsv1.tsv1_model import ts_v1_model

def main():
    """Test if the config file is correctly installed and accessible."""
    print("Testing config file accessibility...")
    
    # Create an instance of the model
    model = ts_v1_model()
    
    # Print the config path
    model.print_config_path()
    
    # Validate the config file
    try:
        model.validate_config()
        print("✅ Config file is valid and correctly installed.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
        # Check if the directory exists
        config_dir = os.path.dirname(model.config_path)
        if not os.path.exists(config_dir):
            print(f"Directory does not exist: {config_dir}")
            print("You may need to create this directory and copy the config file.")
        
        # Provide suggestion for installation
        print("\nSuggestions to fix:")
        print("1. Check if the package is installed correctly")
        print("2. Ensure the config directory exists at:", os.path.dirname(model.config_path))
        print("3. Manually copy the config file to:", model.config_path)
        print("4. If using a package, make sure config files are included in your MANIFEST.in or package_data")
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 