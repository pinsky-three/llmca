use reqwest::{header, Client};

use serde_derive::{Deserialize, Serialize};
use serde_json::json;

use crate::api::ChatCompletionResponse;

#[derive(Default, Debug, Clone)]
pub struct CognitiveUnit {
    pub rule: String,
    pub state: Vec<String>,
    pub position: (usize, usize),
}

#[derive(Debug, Clone, Serialize)]
struct CognitiveUnitInput {
    rule: String,
    state: Vec<String>,
    neighbors: Vec<(String, Vec<String>)>,
}

#[derive(Debug, Clone, Deserialize)]
struct CognitiveUnitOutput {
    next_state: Vec<String>,
}

pub struct CognitiveContext {
    pub client: Box<Client>,
    pub base_api: String,
    pub model_name: String,
    pub secret_key: String,
}

impl CognitiveUnit {
    pub async fn calculate_next_state(
        &self,
        ctx: &CognitiveContext,
        neighbors: Vec<(String, Vec<String>)>,
    ) -> Vec<String> {
        self.llm_model_call(ctx, neighbors).await
    }

    pub async fn llm_model_call(
        &self,
        ctx: &CognitiveContext,
        neighbors: Vec<(String, Vec<String>)>,
    ) -> Vec<String> {
        let system_message = "
        You're a LLM Cognitive Unit and your unique task is to respond with your next 
        state based on the state of your neighbors in json format based on:
        
        #[derive(Debug, Clone, Serialize)]
        struct CognitiveUnitInput {
            rule: String,
            state: Vec<String>, // json sequence of states as any kind of string
            neighbors: Vec<(String, Vec<String>)>, // json sequence of (name, state)
        } 

        #[derive(Debug, Clone, Deserialize)]
        struct CognitiveUnitOutput {
            next_state: Vec<String>, // json sequence of states as any kind of string
        }

        example input:
        {
            \"rule\": \"You're a cellular automaton with game of life behavior. 
            Response with your next state based on the state of your neighbors.
            Don't response with explanations or nothing else, only your state.
            Always respond with a single state and ignoring the rest.\",
            \"state\": [\"0\"],
            \"neighbors\": [
                [\"n_0\", [\"1\"]],
                [\"n_1\", [\"0\"]],
                [\"n_2\", [\"1\"]],
                [\"n_3\", [\"0\"]],
                [\"n_4\", [\"1\"]],
                [\"n_5\", [\"0\"]],
                [\"n_6\", [\"0\"]],
                [\"n_7\", [\"0\"]]
            ]
        }

        example output:
        {
            \"next_state\": [\"1\"]
        }

        Be careful with your response, it should be a valid json with the next_state field.
        Take care of trailing characters, spaces, and new lines.
        "
        .to_string();

        let input_payload = serde_json::to_string_pretty(&CognitiveUnitInput {
            rule: self.rule.clone(),
            state: self.state.clone(),
            neighbors,
        })
        .unwrap();

        let res = Self::generic_chat_completion(ctx, system_message, input_payload).await;

        serde_json::from_str::<CognitiveUnitOutput>(
            &res.unwrap()
                .choices
                .first()
                .unwrap()
                .clone()
                .message
                .content,
        )
        .map(|o| o.next_state)
        .unwrap_or_else(|err| {
            println!("Error in LLM model call ({:?}): {:?}", self.position, err);

            self.state.clone()
        })
    }

    async fn generic_chat_completion(
        ctx: &CognitiveContext,
        system_message: String,
        user_message: String,
    ) -> Result<ChatCompletionResponse, Box<dyn std::error::Error>> {
        // let base_api = ctx.map(|c| c.base_api.clone()).unwrap_or_else(|| {
        //     env::var("OPENAI_API_URL").unwrap_or("http://localhost:11434/v1".to_string())
        // });

        // let model_name = ctx
        //     .map(|c| c.model_name.clone())
        //     .unwrap_or_else(|| env::var("OPENAI_MODEL_NAME").unwrap_or("phi3".to_string()));

        // let secret_key = ctx
        //     .map(|c| c.secret_key.clone())
        //     .unwrap_or_else(|| env::var("OPENAI_API_KEY").unwrap_or("ollama".to_string()));

        let mut headers = header::HeaderMap::new();

        headers.insert("Content-Type", "application/json".parse().unwrap());
        headers.insert(
            "Authorization",
            format!("Bearer {}", ctx.secret_key).parse().unwrap(),
        );

        let body = json!({
            "model": ctx.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        });

        // match ctx {
        //     Some(context) => {

        //     }
        //     None => {
        //         let client = reqwest::blocking::Client::builder()
        //             .timeout(Duration::from_secs(120))
        //             .redirect(reqwest::redirect::Policy::none())
        //             .build()
        //             .unwrap();

        //         let res = client
        //             // .post("https://openrouter.ai/api/v1/chat/completions")
        //             .post(format!("{api_url}/chat/completions"))
        //             .headers(headers)
        //             .body(body.to_string())
        //             .send()
        //             .unwrap();

        //         let parsed_res = res.json::<ChatCompletionResponse>().unwrap();

        //         Ok(parsed_res)
        //     }
        // }

        let res = ctx
            .client
            .post(format!("{}/chat/completions", ctx.base_api))
            .headers(headers)
            .body(body.to_string())
            .send()
            .await
            .unwrap();

        let parsed_res = res.json::<ChatCompletionResponse>().await.unwrap();

        Ok(parsed_res)
    }
}
