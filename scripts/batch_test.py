import os
import requests

# 🛠️ CONFIGURATION
BASE_URL = 'http://127.0.0.1:5000/detect_ripeness'
TEST_IMAGES_DIR = 'C:\\Users\\carro\\Downloads\\for testing purposes' # <--- CHANGE THIS PATH!

def run_batch_test():
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"❌ Error: Folder not found: {TEST_IMAGES_DIR}")
        return

    files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🚀 Found {len(files)} images. Starting batch test...\n")

    for i, filename in enumerate(files):
        filepath = os.path.join(TEST_IMAGES_DIR, filename)
        
        try:
            with open(filepath, 'rb') as img:
                # Send the POST request just like the browser does
                print(f"[{i+1}/{len(files)}] Uploading {filename}...", end=" ")
                response = requests.post(BASE_URL, files={'file': img})
                
                # Check if the server accepted it (200 OK)
                if response.status_code == 200:
                    print("✅ Success!")
                else:
                    print(f"❌ Failed (Status: {response.status_code})")
                    
        except Exception as e:
            print(f"⚠️ Error sending file: {e}")

    print("\n🏁 Batch test complete! Check your terminal logs for the detailed AI breakdown.")

if __name__ == "__main__":
    run_batch_test()