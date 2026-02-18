import argparse
import shutil
import csv
from pathlib import Path
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Harvest PDF reports from multiple output roots into a centralized analysis folder."
    )
    parser.add_argument(
        "--source-dirs", 
        nargs='+', 
        required=True,
        help="List of source directories to scan (e.g. ../output_H100 ../output_L40)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./consolidated_reports",
        help="Where to save the renamed PDFs"
    )
    return parser.parse_args()

def generate_unique_name(origin_tag, relative_path):
    """
    Converts a deep directory path into a flat, readable filename.
    Example: 
      Input:  origin="H100", path="diffusion/cifar/seed-42/server-L128/silos/2026.../run1"
      Output: H100__diffusion__cifar__seed-42__server-L128__silos__run1.pdf
    """
    # Replace path separators with double underscores for readability
    clean_path = str(relative_path).replace("/", "__").replace("\\", "__")
    
    # Remove the timestamp folder parts to shorten the name (optional, but recommended)
    # This regex removes parts that look like dates/times if you want shorter names, 
    # but keeping them ensures uniqueness. We will keep them for safety.
    
    filename = f"{origin_tag}__{clean_path}.pdf"
    return filename

def main():
    args = parse_arguments()
    target_root = Path(args.output_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    # Initialize summary list for CSV
    summary_data = []
    
    print(f"🚀 Starting Report Collection...")
    print(f"📂 Output Directory: {target_root.resolve()}")

    total_copied = 0

    for source in args.source_dirs:
        source_path = Path(source).resolve()
        
        if not source_path.exists():
            print(f"⚠️  Warning: Source path does not exist: {source_path}")
            continue

        # Create a tag from the folder name (e.g., "federatedfactory_output_H100" -> "H100")
        # Heuristic: split by underscore and take the last part, or use full name
        origin_tag = source_path.name.split('_')[-1] if '_' in source_path.name else source_path.name
        
        print(f"🔍 Scanning source: {source_path} (Tag: {origin_tag})")

        # Recursive search for report.pdf
        for report_file in source_path.rglob("report.pdf"):
            experiment_dir = report_file.parent
            
            # Get path relative to the source root (e.g. diffusion/cifar/seed-42/...)
            relative_path = experiment_dir.relative_to(source_path)
            
            # 1. ORGANIZE BY DATASET (First folder in structure usually)
            # This creates subfolders in output like: consolidated_reports/cifar/
            try:
                # Assuming path structure: model / dataset / ...
                # We grab the second part as dataset, or first if structure differs
                parts = relative_path.parts
                dataset_group = parts[1] if len(parts) > 1 else "misc"
            except IndexError:
                dataset_group = "misc"

            group_dir = target_root / dataset_group
            group_dir.mkdir(exist_ok=True)

            # 2. GENERATE NEW FILENAME
            new_filename = generate_unique_name(origin_tag, relative_path)
            destination = group_dir / new_filename

            # 3. COPY FILE
            try:
                shutil.copy2(report_file, destination)
                total_copied += 1
                
                # Add to summary data
                summary_data.append({
                    "origin": origin_tag,
                    "dataset": dataset_group,
                    "original_path": str(experiment_dir),
                    "new_filename": new_filename,
                    "collected_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"❌ Error copying {relative_path}: {e}")

    # 4. GENERATE CSV INDEX
    if summary_data:
        csv_path = target_root / "report_index.csv"
        keys = summary_data[0].keys()
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary_data)
        print(f"\n📄 Index generated: {csv_path}")

    print(f"\n✅ Done! Collected {total_copied} reports into '{target_root}'.")

if __name__ == "__main__":
    main()
