#[cfg(feature = "ssr")]
#[tokio::main]
async fn main() {
    use axum::Router;
    // use dynamical_system::{
    //     system::space::build_lattice_with_memory, system::unit_next::CognitiveUnitPair,
    // };
    use leptos::*;
    use leptos_axum::{generate_route_list, LeptosRoutes};
    use life::app::*;
    use life::fileserv::file_and_error_handler;

    // use axum::extract::FromRef;
    // use dynamical_system::system::space::CognitiveSpaceWithMemory;

    // // #[derive(FromRef, Debug, Clone)]
    // // pub struct AppState {
    // //     // pub leptos_options: LeptosOptions,
    // //     // pub space: CognitiveSpaceWithMemory,
    // //     pub space_serialized: String,
    // // }

    // let (n, m) = (10, 10);

    // let rule = "you represent a color that evokes the sadness".to_string();

    // let space = build_lattice_with_memory(n, m, 4, |_pos| CognitiveUnitPair {
    //     rule: rule.clone(),
    //     state: "#bababa".to_string(),
    // });

    // let space_serialized = space.serialize_in_pretty_json();

    // let state = AppState { space_serialized };
    // Setting get_configuration(None) means we'll be using cargo-leptos's env values
    // For deployment these variables are:
    // <https://github.com/leptos-rs/start-axum#executing-a-server-on-a-remote-machine-without-the-toolchain>
    // Alternately a file can be specified such as Some("Cargo.toml")
    // The file would need to be included with the executable when moved to deployment
    let conf = get_configuration(None).await.unwrap();
    let leptos_options = conf.leptos_options;
    let addr = leptos_options.site_addr;
    let routes = generate_route_list(App);

    // build our application with a route
    let app = Router::new()
        .leptos_routes(&leptos_options, routes, App)
        .fallback(file_and_error_handler)
        .with_state(leptos_options);
    // .with_state(space);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    logging::log!("listening on http://{}", &addr);
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

#[cfg(not(feature = "ssr"))]
pub fn main() {
    // no client-side main function
    // unless we want this to work with e.g., Trunk for a purely client-side app
    // see lib.rs for hydration function instead
}
