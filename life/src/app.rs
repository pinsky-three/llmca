use crate::error_template::{AppError, ErrorTemplate};
use leptos::*;
use leptos_meta::*;
use leptos_router::*;

#[component]
pub fn App() -> impl IntoView {
    // Provides context that manages stylesheets, titles, meta tags, etc.
    provide_meta_context();

    view! {


        // injects a stylesheet into the document <head>
        // id=leptos means cargo-leptos will hot-reload this stylesheet
        <Stylesheet id="leptos" href="/pkg/life.css"/>

        // sets the document title
        <Title text="Welcome to Leptos"/>

        // content for this welcome page
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

// use serde::Deserialize;

// #[derive(Deserialize, Debug)]
// struct MyQuery {
// foo: String,
// }

#[server]
pub async fn server_function_example(title: String) -> Result<String, ServerFnError> {
    use axum::{extract::Query, http::Method};
    use leptos_axum::extract;

    let method: Method = extract().await.unwrap();
    // println!("method: {:?}", method);

    Ok(method.to_string())
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
        </Suspense>
    }
}
