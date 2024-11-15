use std::path::PathBuf;

pub struct LifeManager {
    root_folder: PathBuf,
}

impl Default for LifeManager {
    fn default() -> Self {
        let root_folder = PathBuf::from(".life");

        if !root_folder.exists() {
            std::fs::create_dir(&root_folder).unwrap();
        }

        Self { root_folder }
    }
}

impl LifeManager {
    pub fn register_entity(&self, id: String) -> PathBuf {
        let entity_folder = self.root_folder.join(&id);

        std::fs::create_dir(&entity_folder).unwrap();

        entity_folder
    }

    pub fn list_entities(&self) -> Vec<PathBuf> {
        std::fs::read_dir(&self.root_folder)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect()
    }
}
