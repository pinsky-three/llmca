use std::path::PathBuf;

use super::entity::Entity;
use crate::system::space::{load_llm_resolvers_from_toml, LLMResolver};
use serde_derive::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LifeManager {
    root_folder: PathBuf,
    loaded_entities: Vec<Entity>,
    resolvers: Vec<LLMResolver>,
}

impl Default for LifeManager {
    fn default() -> Self {
        let root_folder = PathBuf::from(".life");

        if !root_folder.exists() {
            std::fs::create_dir(&root_folder).unwrap();
        }

        let resolvers = load_llm_resolvers_from_toml("resolvers.toml");

        let mut instance = Self {
            root_folder,
            resolvers,
            loaded_entities: vec![],
        };

        instance.list_entities().iter().for_each(|entity_folder| {
            let folder_name = instance.root_folder.join(entity_folder.clone());
            println!("Loading entity from {:?}", folder_name);

            let entity = Entity::open_saved(folder_name, instance.clone());
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

    pub fn get_mut_entity(&mut self, id: &str) -> Option<&mut Entity> {
        self.loaded_entities
            .iter_mut()
            .find(|entity| entity.id() == id)
    }

    pub fn get_all_entities(&self) -> &Vec<Entity> {
        &self.loaded_entities
    }

    pub fn resolvers(&self) -> &Vec<LLMResolver> {
        &self.resolvers
    }

    pub fn set_resolvers(&mut self, resolvers: Vec<LLMResolver>) {
        self.resolvers = resolvers;
    }
}
