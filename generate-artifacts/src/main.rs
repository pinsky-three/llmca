use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io;
// use std::path::Path;

fn main() -> io::Result<()> {
    // Path to the folder containing JSON files
    let folder_path = "./saves/b0fa6b040157e361fbd6ac54e560aaba/1731519735"; // Change this to your folder path
    let output_file = "combined.json";

    let mut combined_data: BTreeMap<String, Value> = BTreeMap::new();

    // Read all files in the folder
    for entry in fs::read_dir(folder_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let filename = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();

            let file_content = fs::read_to_string(&path)?;

            match serde_json::from_str::<Value>(&file_content) {
                Ok(data) => {
                    combined_data.insert(filename, data);
                }
                Err(e) => {
                    eprintln!("Error reading file {}: {}", path.display(), e);
                }
            }
        }
    }

    // Save the combined data to a new JSON file
    let output_json = serde_json::to_string_pretty(&combined_data).unwrap();
    fs::write(output_file, output_json)?;

    println!("Combined data saved to {}", output_file);

    Ok(())
}
