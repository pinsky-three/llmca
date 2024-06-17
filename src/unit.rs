use std::{env, time::Duration};

use reqwest::header;

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

impl CognitiveUnit {
    pub fn calculate_next_state(&self, neighbors: Vec<(String, Vec<String>)>) -> Vec<String> {
        self.llm_model_call(neighbors)
    }

    pub fn llm_model_call(&self, neighbors: Vec<(String, Vec<String>)>) -> Vec<String> {
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
        "
        .to_string();

        let input_payload = serde_json::to_string_pretty(&CognitiveUnitInput {
            rule: self.rule.clone(),
            state: self.state.clone(),
            neighbors,
        })
        .unwrap();

        let res = Self::generic_chat_completion(system_message, input_payload);

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

    fn generic_chat_completion(
        system_message: String,
        user_message: String,
    ) -> Result<ChatCompletionResponse, Box<dyn std::error::Error>> {
        let open_ai_key = env::var("OPENAI_API_KEY").unwrap_or("ollama".to_string());
        let model_name = env::var("OPENAI_MODEL_NAME").unwrap_or("phi3".to_string());

        let mut headers = header::HeaderMap::new();

        headers.insert("Content-Type", "application/json".parse().unwrap());
        headers.insert(
            "Authorization",
            format!("Bearer {}", open_ai_key).parse().unwrap(),
        );

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();

        let body = json!({
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        });

        Ok(client
            // .post("https://openrouter.ai/api/v1/chat/completions")
            .post("http://localhost:11434/v1/chat/completions")
            .headers(headers)
            .body(body.to_string())
            .send()
            .unwrap()
            .json::<ChatCompletionResponse>()?)
    }
}
