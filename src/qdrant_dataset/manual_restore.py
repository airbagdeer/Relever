import tarfile
import os
from pathlib import Path
import shutil
from tqdm import tqdm

# Point these at the downloaded Wikipedia snapshot file and the Qdrant storage dir.
SNAPSHOT_PATH = Path(os.environ.get("QDRANT_SNAPSHOT_PATH", "qdrant_wikipedia.snapshot"))
STORAGE_DIR = Path(os.environ.get("QDRANT_STORAGE_DIR", "qdrant_storage"))
COLLECTION_NAME = "wikipedia_en"
TARGET_DIR = STORAGE_DIR / "collections" / COLLECTION_NAME

def main():
    if not os.environ.get("QDRANT_SNAPSHOT_PATH") or not os.environ.get("QDRANT_STORAGE_DIR"):
        print("Set QDRANT_SNAPSHOT_PATH (the .snapshot file) and QDRANT_STORAGE_DIR "
              "(the Qdrant storage dir mounted into the container) first.")
        return
    if not SNAPSHOT_PATH.exists():
        print(f"Snapshot not found at {SNAPSHOT_PATH}")
        return

    print(f"Target Directory: {TARGET_DIR}")
    
    # Clean up target directory if it exists
    if TARGET_DIR.exists():
        print("Cleaning up existing target directory...")
        shutil.rmtree(TARGET_DIR)
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening snapshot: {SNAPSHOT_PATH}")
    # Get total size for progress bar
    total_size = SNAPSHOT_PATH.stat().st_size
    
    # State tracking
    extracted_bytes = 0
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Restoring")

    try:
        # Open main tarfile
        with tarfile.open(SNAPSHOT_PATH, mode='r|') as main_tar:
            for member in main_tar:
                member_path = Path(member.name)
                
                # Update progress (approximation based on block skips, 
                # tarfile stream iteration doesn't give exact consumed bytes easily
                # so we might just step by member size)
                pbar.update(member.size)

                # Determine destination
                # Snapshot structure: config.json, 0/..., etc.
                # We map this directly to collections/wikipedia_en/
                
                dest_path = TARGET_DIR / member_path
                
                if member.name.endswith(".tar"):
                    # This is a segment tar. We need to unpack it on the fly.
                    # Structure: 0/segments/uuid.tar
                    # Inner structure: snapshot/... 
                    # We want to extract it to: TARGET_DIR/0/segments/uuid/ (renaming 'snapshot' to 'uuid' effectively)
                    
                    segment_uuid = member_path.stem # e.g. "69144f52..."
                    segment_base_dir = dest_path.parent / segment_uuid
                    
                    print(f"\nUnpacking inner tar: {member.name} -> {segment_base_dir}")
                    
                    # Open the inner entry as a stream
                    file_obj = main_tar.extractfile(member)
                    if file_obj:
                        with tarfile.open(fileobj=file_obj, mode='r|') as inner_tar:
                            for inner_member in inner_tar:
                                # We expect members to start with "snapshot/"
                                # e.g. "snapshot/files/version.info"
                                # We want to write to "segment_base_dir/files/version.info"
                                
                                rel_path = Path(inner_member.name)
                                parts = list(rel_path.parts)
                                
                                # Strip 'snapshot' if present
                                if parts and parts[0] == "snapshot":
                                    parts.pop(0)
                                
                                # Check if 'files' is the next part and strip it too if needed
                                # Qdrant expects version.info at the root of the segment dir
                                if parts and parts[0] == "files":
                                    parts.pop(0)
                                    
                                final_rel_path = Path(*parts)
                                
                                final_dest_path = segment_base_dir / final_rel_path
                                
                                if inner_member.isdir():
                                    final_dest_path.mkdir(parents=True, exist_ok=True)
                                else:
                                    final_dest_path.parent.mkdir(parents=True, exist_ok=True)
                                    source = inner_tar.extractfile(inner_member)
                                    if source:
                                        with open(final_dest_path, "wb") as target:
                                            # Copy efficiently
                                            shutil.copyfileobj(source, target)
                                        source.close()
                            
                        file_obj.close()
                    else:
                        print(f"Warning: Could not extract file object for {member.name}")

                else:
                    # Regular file or dir
                    if member.isdir():
                        dest_path.mkdir(parents=True, exist_ok=True)
                    else:
                        # Extract regular file
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        # We use extractfile and write manually to control destination easier
                        source = main_tar.extractfile(member)
                        if source:
                            with open(dest_path, "wb") as target:
                                shutil.copyfileobj(source, target)
                            source.close()
                        
    except Exception as e:
        print(f"\nError encountered: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        print("\nRestoration process finished.")
        print("Please check permissions of the restored files (might need chown for docker user).")

if __name__ == "__main__":
    main()
