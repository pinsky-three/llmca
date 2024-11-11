use crate::error_template::{AppError, ErrorTemplate};
use leptos::*;
use leptos_meta::*;
use leptos_router::*;

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/life.css"/>
        <Title text="Welcome to Leptos"/>
        <Router fallback=|| {
            let mut outside_errors = Errors::default();
            outside_errors.insert_with_default_key(AppError::NotFound);
            view! {
                <ErrorTemplate outside_errors/>
            }
            .into_view()
        }>
            <main>
                <Routes>
                    <Route path="" view=HomePage/>
                </Routes>
            </main>
        </Router>
    }
}

#[server]
pub async fn server_function_example(_title: String) -> Result<String, ServerFnError> {
    // use axum::http::Method;
    // use leptos_axum::extract;
    // use axum::extract::Query;
    use dynamical_system::{
        system::space::build_lattice_with_memory, system::unit_next::CognitiveUnitPair,
    };

    // let method: Method = extract().await.unwrap();

    let (n, m) = (5, 5);

    let rule = "you represent a color that evokes the sadness".to_string();

    let space = build_lattice_with_memory(n, m, 4, |_pos| CognitiveUnitPair {
        rule: rule.clone(),
        state: "#bababa".to_string(),
    });

    // println!("method: {:?}", method);

    let space_serialized = space.serialize_in_pretty_json();

    Ok(space_serialized)
}

/// Renders the home page of your application.
#[component]
fn HomePage() -> impl IntoView {
    // Creates a reactive value to update the button
    let (count, set_count) = create_signal(0);
    let on_click = move |_| set_count.update(|count| *count += 1);

    view! {
        <h1>"Welcome to Leptos!"</h1>
        <button on:click=on_click>"Click Me: " {count}</button>

        <Suspense
            // the fallback will show initially
            // on subsequent reloads, the current child will
            // continue showing
            fallback=move || view! { <p>"Loading..."</p> }
        >
            <button on:click=move |_| {
                spawn_local(async {
                    server_function_example("So much to do!".to_string()).await.unwrap();
                });
            }>
                "Add Todo"
            </button>
            <div>
                <CognitiveUnit rule="".to_string() state="".to_string()/>
            </div>
        </Suspense>
    }
}

#[component]
fn CognitiveUnit(rule: String, state: String) -> impl IntoView {
    let (n, m) = (5, 5);

    view! {
        <div>
            <p>"Cognitive Unit"</p>
            // <p>"This is a cognitive unit."</p>
        </div>
    }
}
