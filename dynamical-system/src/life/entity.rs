use std::{collections::HashSet, fs::read_dir, path::PathBuf, time};

use serde_derive::{Deserialize, Serialize};

use crate::system::{
    space::{build_lattice_with_memory, CognitiveSpaceWithMemory},
    unit_next::CognitiveUnitPair,
};

use super::manager::LifeManager;

#[derive(Debug, Serialize, Deserialize)]
pub struct Entity {
    _id: String,
    artifacts_folder: PathBuf,
    space: CognitiveSpaceWithMemory,
    step: u32,
}

pub enum EntityStatus {
    ComputingStep(u32),
    Idle,
}

impl Entity {
    pub fn new_2d_lattice(
        manager: &LifeManager,
        initial_state: Vec<CognitiveUnitPair>,
        size: (usize, usize),
        temporal_memory_size: usize,
    ) -> Self {
        let space = build_lattice_with_memory(size.0, size.1, temporal_memory_size, |(x, y)| {
            initial_state[(x + y * size.1) % initial_state.len()].clone()
        });

        let space_hash = format!("{:x}", md5::compute(space.serialize_in_pretty_json()));

        let timestamp = time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let id = format!("{}-{}", space_hash, timestamp);

        let artifacts_folder = manager.register_entity(id.clone());

        let step = 0;

        let ent = Self {
            _id: id,
            artifacts_folder,
            space,
            step,
        };

        ent.save_serialized();

        ent
    }

    pub fn id(&self) -> &str {
        &self._id
    }

    pub fn save_serialized(&self) {
        let ser_space = self.space.serialize_in_pretty_json();

        std::fs::write(
            self.artifacts_folder.join(format!("{}.json", self.step)),
            ser_space,
        )
        .unwrap();
    }

    pub fn open_saved(artifacts_folder: PathBuf) -> Self {
        let folder = read_dir(&artifacts_folder).unwrap();

        let steps = folder
            .map(|entry| {
                let entry = entry.unwrap();
                entry.path()
            })
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("json"))
            .map(|path| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string()
            })
            .filter_map(|entry| entry.parse::<u32>().ok())
            .collect::<Vec<_>>();

        let last_step = steps.iter().max().unwrap();
        let last_step_path = artifacts_folder.join(format!("{}.json", last_step));

        let json = std::fs::read_to_string(last_step_path).unwrap();

        let space = CognitiveSpaceWithMemory::load_from_json(&json);

        let id = artifacts_folder
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        Self {
            _id: id,
            artifacts_folder,
            space,
            step: *last_step,
        }
    }

    pub fn current_step(&self) -> u32 {
        self.step
    }

    pub fn space(&self) -> &CognitiveSpaceWithMemory {
        &self.space
    }

    pub fn calculate_unique_states(&self) -> HashSet<String> {
        let all_states = self
            .space
            .get_units()
            .iter()
            .map(|u| u.memory.last().unwrap().state.clone())
            .collect::<Vec<_>>();

        let unique_states = all_states.iter().cloned().collect::<HashSet<_>>();

        unique_states
    }

    pub async fn evolve(&mut self) {
        self.space.distributed_step().await;
        self.step += 1;

        self.save_serialized();
    }
}
