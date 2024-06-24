use reqwest::{header, Client};

use serde_derive::{Deserialize, Serialize};
use serde_json::json;

use crate::api::ChatCompletionResponse;

#[derive(Default, Debug, Clone)]
pub struct CognitiveUnit {
    pub rule: String,
    pub state: Vec<String>,
    pub position: (usize, usize),
    pub feedback: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CognitiveUnitInput {
    rule: String,
    state: Vec<String>,
    neighbors: Vec<(String, Vec<String>)>,
    feedback: Option<String>,
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

pub struct LLMComputationResult {
    pub calculated_state: Vec<String>,
    pub feedback: Option<String>,
}

impl CognitiveUnit {
    pub async fn calculate_next_state(
        &self,
        ctx: &CognitiveContext,
        neighbors: Vec<(String, Vec<String>)>,
    ) -> LLMComputationResult {
        self.llm_model_call(ctx, neighbors).await
    }

    pub async fn llm_model_call(
        &self,
        ctx: &CognitiveContext,
        neighbors: Vec<(String, Vec<String>)>,
    ) -> LLMComputationResult {
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

        let input_payload = serde_json::to_string_pretty(&&CognitiveUnitInput {
            rule: self.rule.clone(),
            state: self.state.clone(),
            feedback: self.feedback.clone(),
            neighbors,
        })
        .unwrap();

        let res = Self::generic_chat_completion(ctx, system_message, input_payload).await;

        match serde_json::from_str::<CognitiveUnitOutput>(
            &res.unwrap()
                .choices
                .first()
                .unwrap()
                .clone()
                .message
                .content,
        ) {
            Ok(output) => LLMComputationResult {
                calculated_state: output.next_state,
                feedback: None,
            },
            Err(err) => LLMComputationResult {
                calculated_state: self.state.clone(),
                feedback: Some(format!("Error from Learning Machine: {:?}", err)),
            },
        }
    }

    async fn generic_chat_completion(
        ctx: &CognitiveContext,
        system_message: String,
        user_message: String,
    ) -> Result<ChatCompletionResponse, Box<dyn std::error::Error>> {
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
