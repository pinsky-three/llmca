use dynamical_system::life::manager::LifeManager;
use poem::{listener::TcpListener, Route};
use poem_openapi::{
    param::Query,
    payload::{Json, PlainText},
    OpenApi, OpenApiService,
};
use serde_json::{json, Value};

struct Api {
    life_manager: LifeManager,
}

#[OpenApi]
impl Api {
    #[oai(path = "/hello", method = "get")]
    async fn index(&self, name: Query<Option<String>>) -> PlainText<String> {
        match name.0 {
            Some(name) => PlainText(format!("hello, {}!", name)),
            None => PlainText("hello!".to_string()),
        }
    }

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
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    let api_service = OpenApiService::new(
        Api {
            life_manager: LifeManager::default(),
        },
        "Hello World",
        "1.0",
    )
    .server("http://localhost:3000/api");

    let ui = api_service.swagger_ui();
    let app = Route::new().nest("/api", api_service).nest("/", ui);

    poem::Server::new(TcpListener::bind("0.0.0.0:3000"))
        .run(app)
        .await
}
