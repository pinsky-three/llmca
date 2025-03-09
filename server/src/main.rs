use std::sync::Arc;

use dynamical_system::life::manager::LifeManager;
use poem::{listener::TcpListener, middleware::Cors, EndpointExt, Route};
use poem_openapi::{param::Path, payload::Json, OpenApi, OpenApiService};
use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug)]
struct Api {
    life_manager: Arc<LifeManager>,
}

#[derive(Deserialize, Serialize)]
pub struct CognitiveUnitComplex {
    pub rule: Option<String>,
    pub state: Option<String>,          // in json format
    pub neighbors: Option<Vec<String>>, // in json format
    pub feedback: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct InteractionPayload {
    i: usize,
    j: usize,
    update_unit: CognitiveUnitComplex,
}

#[OpenApi]
impl Api {
    #[oai(path = "/life", method = "get")]
    async fn life(&self) -> Json<Value> {
        let entities = self.life_manager.list_entities();

        Json(json!({ "entities": entities }))
    }

    #[oai(path = "/life", method = "post")]
    async fn create_life(&self, id: Json<String>) -> Json<Value> {
        let entity = self.life_manager.register_entity(id.0);

        Json(json!({ "entity": entity }))
    }

    #[oai(path = "/life/:id", method = "get")]
    async fn get_life(&self, id: Path<String>) -> Json<Value> {
        let entity = self.life_manager.get_entity(&id.0);

        match entity {
            Some(entity) => Json(json!({ "entity": entity })),
            None => Json(json!({ "error": "entity not found" })),
        }
    }

    #[oai(path = "/entity/:id/evolve", method = "post")]
    async fn evolve_simulation(&self, id: Path<String>) -> Json<Value> {
        let mut life_manager = self.life_manager.clone();

        let life_manager = Arc::get_mut(&mut life_manager).unwrap();

        match life_manager.get_mut_entity(&id.0) {
            Some(entity) => {
                entity.evolve().await;
                Json(json!({ "status": "done" }))
            }
            None => Json(json!({ "error": "entity not found" })),
        }
    }

    #[oai(path = "/entity/:id/interact", method = "post")]
    async fn interact_simulation(
        &self,
        id: Path<String>,
        payload: Json<Vec<String>>,
    ) -> Json<Value> {
        let mut life_manager = self.life_manager.clone();

        let life_manager = Arc::get_mut(&mut life_manager).unwrap();

        let entity = life_manager.get_mut_entity(&id.0).unwrap();

        // entity.space().set_unit(i, j, unit);

        println!("Payload: {:?}", payload);
        println!("Entity: {:?}", entity);

        todo!()

        // match a.get_mut_entity(&id.0) {
        //     Some(entity) => {
        //         entity.interact(payload.0).await;
        //         Json(json!({ "status": "done" }))
        //     }
        //     None => Json(json!({ "error": "entity not found" })),
        // }
    }
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    let api_service = OpenApiService::new(
        Api {
            life_manager: Arc::new(LifeManager::default()),
        },
        "LLMCA API",
        "1.0",
    )
    .server("http://localhost:8000/api");

    // let ui = api_service.swagger_ui();
    let app = Route::new().nest("/api", api_service).with(
        Cors::new()
            .allow_origin("*")
            .allow_methods(vec!["GET", "POST", "OPTIONS"])
            .allow_headers(vec!["Content-Type", "Authorization"]),
    );

    poem::Server::new(TcpListener::bind("0.0.0.0:8000"))
        .run(app)
        .await
}
