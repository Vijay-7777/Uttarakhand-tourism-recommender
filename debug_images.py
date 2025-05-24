import pandas as pd
import os

# Load the CSV file
df = pd.read_csv('data/uttarakhand_places.csv')

print("=== CSV FILE DEBUG INFO ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 3 rows:")
print(df.head(3))

print("\n=== IMAGE FILES DEBUG INFO ===")

# Check if static/images directory exists
images_dir = 'static/images'
if os.path.exists(images_dir):
    print(f"✅ {images_dir} directory exists")
    
    # List all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
    print(f"Found {len(image_files)} image files:")
    for img in sorted(image_files):
        print(f"  - {img}")
    
    print("\n=== IMAGE MAPPING CHECK ===")
    
    # Try to find the image column in CSV
    image_column = None
    for col in df.columns:
        if 'image' in col.lower() or 'url' in col.lower():
            image_column = col
            break
    
    if image_column:
        print(f"Image column found: '{image_column}'")
        unique_images = df[image_column].dropna().unique()
        print(f"Unique image references in CSV: {len(unique_images)}")
        
        print("\nChecking if CSV images exist in folder:")
        missing_images = []
        for img_ref in unique_images:
            if img_ref and isinstance(img_ref, str):
                if img_ref in image_files:
                    print(f"  ✅ {img_ref}")
                else:
                    print(f"  ❌ {img_ref} (MISSING)")
                    missing_images.append(img_ref)
        
        if missing_images:
            print(f"\n🚨 {len(missing_images)} images are referenced in CSV but missing from folder:")
            for img in missing_images:
                print(f"  - {img}")
    else:
        print("❌ No image column found in CSV")
        
    # Check for alternative naming patterns
    print("\n=== ALTERNATIVE IMAGE NAMING CHECK ===")
    if 'Name' in df.columns or 'name' in df.columns:
        name_col = 'Name' if 'Name' in df.columns else 'name'
        for _, row in df.head(10).iterrows():  # Check first 10 places
            place_name = row[name_col]
            # Generate possible image names
            possible_names = [
                f"{place_name.lower().replace(' ', '_')}.jpg",
                f"{place_name.lower().replace(' ', '_').replace(',', '')}.jpg",
                f"{place_name.lower().replace(' ', '-')}.jpg",
                f"{place_name.lower().replace(' ', '')}.jpg",
            ]
            
            found = False
            for possible_name in possible_names:
                if possible_name in image_files:
                    print(f"  ✅ {place_name} -> {possible_name}")
                    found = True
                    break
            
            if not found:
                print(f"  ❌ {place_name} -> No matching image found")
                print(f"      Tried: {', '.join(possible_names)}")
    
else:
    print(f"❌ {images_dir} directory does not exist!")
    print("Please create the directory and add your images.")

print("\n=== RECOMMENDATIONS ===")

if not os.path.exists(images_dir):
    print("1. Create the 'static/images/' directory")
    print("2. Add your image files to this directory")
elif len(image_files) == 0:
    print("1. Add image files to the 'static/images/' directory")
    print("2. Make sure they match the names in your CSV file")
else:
    print("1. Check the console output when running your Flask app")
    print("2. Look at the browser's developer console for image loading errors")
    print("3. The updated Flask app includes debugging information")

print("\n=== SAMPLE IMAGE RENAMING SCRIPT ===")
print("If your images don't match CSV names, you can rename them using:")
print("""
import os
import pandas as pd

df = pd.read_csv('data/uttarakhand_places.csv')
images_dir = 'static/images'

# Example: Rename images to match place names
for _, row in df.iterrows():
    place_name = row['Name']  # Adjust column name as needed
    target_name = f"{place_name.lower().replace(' ', '_').replace(',', '')}.jpg"
    print(f"Rename an image to: {target_name}")
""")