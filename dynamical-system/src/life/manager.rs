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
            let entity = Entity::open_saved(entity_folder.clone());
            instance.loaded_entities.push(entity);
        });

        instance
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

    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.loaded_entities.iter().find(|entity| entity.id() == id)
    }
}
