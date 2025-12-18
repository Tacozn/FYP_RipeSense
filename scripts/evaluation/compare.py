import hashlib

def get_file_hash(file_path):
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return "File not found!"

# --- UPDATE THESE PATHS ---
# Make sure these match your actual folder names!
old_file = "weights_1/best.pt"   # The old file
new_file = "weights_new/best.pt" # The new file you just downloaded

hash1 = get_file_hash(old_file)
hash2 = get_file_hash(new_file)

print(f"Old File Hash: {hash1}")
print(f"New File Hash: {hash2}")

if hash1 == hash2:
    print(">>> MATCH: These are the EXACT same file. You are using the old model.")
else:
    print(">>> DIFFERENT: These are different files. Your new model is active.")