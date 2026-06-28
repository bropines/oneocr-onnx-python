# /// script
# dependencies = [
#   "requests>=2.28.0",
#   "packaging>=23.0",
# ]
# ///
import os
import re
import sys
import zipfile
import requests
import shutil
from packaging.version import parse as parse_version

# Store settings
STORE_URL = "https://store.rg-adguard.net/api/GetFiles"
PRODUCT_URL = "https://apps.microsoft.com/detail/9mz95kl8mr0l" # ScreenSketch (Snipping Tool)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://store.rg-adguard.net",
    "Referer": "https://store.rg-adguard.net/",
    "Content-Type": "application/x-www-form-urlencoded"
}

def get_latest_bundle_url():
    print("Querying Microsoft Store API for latest Snipping Tool package...")
    data = {
        "type": "url",
        "url": PRODUCT_URL,
        "ring": "Retail",
        "lang": "en-US"
    }
    
    response = requests.post(STORE_URL, data=data, headers=headers, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to query store API: HTTP {response.status_code}")
    
    html = response.text
    # Find all msixbundle links
    pattern = r'href="([^"]+)"[^>]*>(Microsoft\.ScreenSketch_([\d\.]+)_neutral_~_8wekyb3d8bbwe\.msixbundle)</a>'
    matches = re.findall(pattern, html)
    
    if not matches:
        raise RuntimeError("No Microsoft.ScreenSketch .msixbundle packages found in the response.")
    
    bundles = []
    for url, filename, version_str in matches:
        bundles.append({
            'url': url,
            'filename': filename,
            'version': parse_version(version_str),
            'version_str': version_str
        })
    
    # Sort by version descending
    bundles.sort(key=lambda x: x['version'], reverse=True)
    latest = bundles[0]
    print(f"Latest package version found: {latest['version_str']}")
    return latest['url'], latest['filename']

def download_file(url, output_path):
    print(f"Downloading package file: {os.path.basename(output_path)}...")
    response = requests.get(url, stream=True, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download package: HTTP {response.status_code}")
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB chunks
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\rProgress: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end='', flush=True)
                else:
                    print(f"\rProgress: {downloaded / (1024 * 1024):.1f} MB", end='', flush=True)
    print("\nDownload complete.")

def extract_oneocr_files(bundle_path, target_arch="x64", output_dir=None):
    if output_dir is None:
        # Default to a local 'bin' directory inside tmp/OneOCR_Deobfuscated
        output_dir = os.path.join(os.getcwd(), 'tmp', 'OneOCR_Deobfuscated', 'bin')
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extraction directory: {output_dir}")
    
    temp_dir = "temp_oneocr_extract"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # Step 1: Open bundle and locate the architecture msix
        print("Searching bundle for architecture msix...")
        target_msix_name = None
        with zipfile.ZipFile(bundle_path, 'r') as bundle:
            for name in bundle.namelist():
                if target_arch in name and name.endswith('.msix'):
                    target_msix_name = name
                    print(f"Found inner MSIX: {name}")
                    bundle.extract(name, temp_dir)
                    break
        
        if not target_msix_name:
            raise RuntimeError(f"Could not find any MSIX for architecture {target_arch} inside the bundle.")
        
        msix_path = os.path.join(temp_dir, target_msix_name)
        
        # Step 2: Open architecture msix and search for oneocr files
        # Use a set to track which files we already have so duplicates from
        # sub-directories (e.g. SnippingToolSandbox/, SnippingTool/, etc.) don't
        # overwrite the first - correct - copy we find.
        print("Searching MSIX for oneocr files...")
        extracted_files = []
        target_filenames = ["oneocr.dll", "oneocr.onemodel", "onnxruntime.dll"]
        already_extracted = set()
        
        with zipfile.ZipFile(msix_path, 'r') as msix:
            for name in msix.namelist():
                basename = os.path.basename(name).lower()
                if basename in target_filenames and basename not in already_extracted:
                    print(f"Extracting {name} -> {output_dir}")
                    target_path = os.path.join(output_dir, os.path.basename(name))
                    with msix.open(name) as source, open(target_path, 'wb') as dest:
                        shutil.copyfileobj(source, dest)
                    extracted_files.append(target_path)
                    already_extracted.add(basename)
                    # Stop early once we have all 3 targets
                    if len(already_extracted) == len(target_filenames):
                        break
        
        if len(extracted_files) < len(target_filenames):
            missing = set(target_filenames) - already_extracted
            print(f"Warning: Missing files from package: {missing}")
        else:
            print("\nSuccessfully extracted all 3 OneOCR components!")
            print(f"Files are ready in: {output_dir}")
            
    finally:
        # Clean up temp folder
        print("Cleaning up temporary directories...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    try:
        url, filename = get_latest_bundle_url()
        temp_bundle = "temp_sketch_download.msixbundle"
        
        # Download the bundle
        download_file(url, temp_bundle)
        
        # Extract files
        extract_oneocr_files(temp_bundle, target_arch="x64")
        
        # Clean up downloaded bundle
        if os.path.exists(temp_bundle):
            os.remove(temp_bundle)
            
        print("\nAll done! You can now run the decryption script.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
