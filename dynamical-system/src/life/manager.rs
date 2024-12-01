use std::path::PathBuf;

use super::entity::Entity;

pub struct LifeManager {
    root_folder: PathBuf,
    loaded_entities: Vec<Entity>,
}

impl Default for LifeManager {
    fn default() -> Self {
        let root_folder = PathBuf::from(".life");

        if !root_folder.exists() {
            std::fs::create_dir(&root_folder).unwrap();
        }

        let mut instance = Self {
            root_folder,
            loaded_entities: vec![],
        };

        instance.list_entities().iter().for_each(|entity_folder| {
            let folder_name = instance.root_folder.join(entity_folder.clone());
            println!("Loading entity from {:?}", folder_name);

            let entity = Entity::open_saved(folder_name);
            instance.loaded_entities.push(entity);
        });

        instance
    }
}

impl LifeManager {
    pub fn root_folder(&self) -> &PathBuf {
        &self.root_folder
    }

    pub fn register_entity(&self, id: String) -> PathBuf {
        let entity_folder = self.root_folder.join(&id);

        std::fs::create_dir(&entity_folder).unwrap();

        entity_folder
    }

    pub fn list_entities(&self) -> Vec<String> {
        std::fs::read_dir(&self.root_folder)
            .unwrap()
            .map(|entry| {
                entry
                    .unwrap()
                    .path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
            })
            .collect()
    }

    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.loaded_entities.iter().find(|entity| entity.id() == id)
    }

    pub fn get_all_entities(&self) -> &Vec<Entity> {
        &self.loaded_entities
    }
}
