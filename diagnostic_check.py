import os
import sys

def check_path(path):
    print(f"Checking: {path}")
    if os.path.exists(path):
        print(f"  EXISTS")
        if os.path.isfile(path):
            print(f"  IS_FILE")
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(100)
                    print(f"  READ_OK (First 100 bytes): {content!r}")
            except Exception as e:
                print(f"  READ_ERROR: {e}")
        elif os.path.isdir(path):
            print(f"  IS_DIR")
            try:
                items = os.listdir(path)
                print(f"  LIST_OK: Found {len(items)} items")
                print(f"  Items: {items[:5]}...")
            except Exception as e:
                print(f"  LIST_ERROR: {e}")
    else:
        print(f"  NOT_FOUND")

base_dir = r"C:\GeoSpec"
sprint_dir = r"C:\GeoSpec\geospec_sprint"

targets = [
    r"C:\GeoSpec",
    r"C:\GeoSpec\geospec_sprint"
]

print(f"CWD: {os.getcwd()}")
print("-" * 40)

for t in targets:
    print(f"Listing directory: {t}")
    try:
        if os.path.exists(t):
            items = os.listdir(t)
            print(f"  Found {len(items)} items")
            for item in items:
                if "geospec_algorithms" in item or "env" in item or "wsl" in item:
                    print(f"  MATCH: '{item}' (len={len(item)})")
        else:
            print("  DIR_NOT_FOUND")
    except Exception as e:
        print(f"  ERROR: {e}")
    print("-" * 40)
