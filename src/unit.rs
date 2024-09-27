use reqwest::{header, Client};
use schemars::{schema_for, JsonSchema};

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

#[derive(Default, Debug, Clone)]
pub struct CognitiveSubstrateUnit {
    prompt: String,
    pub max_size: u64,
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

#[derive(Debug, Clone, Serialize, JsonSchema)]
struct CognitiveUnitInput {
    rule: String,
    state: Vec<String>,
    neighbors: Vec<(String, Vec<String>)>,
    feedback: Option<String>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct CognitiveUnitOutput {
    next_state: Vec<String>,
}

impl CognitiveContext {
    async fn generic_chat_completion(
        &self,
        system_message: String,
        _user_message: String,
    ) -> Result<ChatCompletionResponse, Box<dyn std::error::Error>> {
        let mut headers = header::HeaderMap::new();

        headers.insert("Content-Type", "application/json".parse().unwrap());
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.secret_key).parse().unwrap(),
        );

        let body = json!({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                // {"role": "user", "content": _user_message}
            ]
        });

        // println!("body: {}", body);

        let res = self
            .client
            .post(format!("{}/chat/completions", self.base_api))
            .headers(headers)
            .body(body.to_string())
            .send()
            .await
            .unwrap();

        let parsed_res = res.json::<ChatCompletionResponse>().await.unwrap();

        // println!("parsed_res: {:?}", parsed_res);

        Ok(parsed_res)
    }
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
        let unit_input_json_schema =
            serde_json::to_string_pretty(&schema_for!(CognitiveUnitInput)).unwrap();

        let unit_output_json_schema =
            serde_json::to_string_pretty(&schema_for!(CognitiveUnitOutput)).unwrap();

        let system_message = format!(
            "
        You're a LLM Cognitive Unit and your unique task is to respond with your next 
        state based on the state of your neighbors in json format based on:
        

        input_json_schema:
        ```json
        {unit_input_json_schema}
        ```

        output_json_schema:
        ```json
        {unit_output_json_schema}
        ```

        example input: 
        {{
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
        }}

        example output:
        {{
            \"next_state\": [\"1\"]
        }}

        You can use the state array to store valuable information about the unit and its neighbors.
        Each value in your state array is called a channel and can be used to store different types of information.

        Be careful with your response, it should be a valid json with the next_state field.
        Take care of trailing characters, spaces, and new lines.
        "
        );

        let input_payload = serde_json::to_string_pretty(&&CognitiveUnitInput {
            rule: self.rule.clone(),
            state: self.state.clone(),
            feedback: self.feedback.clone(),
            neighbors,
        })
        .unwrap();

        let res = ctx
            .generic_chat_completion(system_message, input_payload)
            .await;

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
}

impl CognitiveSubstrateUnit {
    pub fn new(prompt: String, max_size: u64) -> Self {
        Self { prompt, max_size }
    }

    pub fn get_prompt(&self) -> String {
        self.prompt
            .chars()
            .take(self.max_size as usize)
            .collect::<String>()
    }

    pub async fn compute(&self, ctx: &CognitiveContext) -> String {
        // let system_message = self.prompt.clone();
        let system_message = self.get_prompt();
        let user_message = format!(
            "current_prompt: {}\nmax_size: {}",
            system_message, self.max_size
        );

        let parsed_res = ctx
            .generic_chat_completion(system_message, user_message)
            .await
            .unwrap();

        parsed_res
            .choices
            .first()
            .unwrap()
            .message
            .content
            .trim()
            .to_string()
    }

    pub async fn cross_with(
        &self,
        ctx: &CognitiveContext,
        other: &CognitiveSubstrateUnit,
    ) -> (String, String) {
        let system_message_1 = self.get_prompt();
        let system_message_2 = other.get_prompt();

        let system_message = format!("{}\n{}", system_message_1, system_message_2);

        let user_message = format!(
            "current_prompt: {}\nmax_size: {}",
            system_message, self.max_size
        );

        let parsed_res = ctx
            .generic_chat_completion(system_message, user_message)
            .await
            .unwrap();

        // println!("parsed_res: {:?}", parsed_res);

        let mut result = parsed_res
            .choices
            .first()
            .unwrap()
            .message
            .content
            .split('\n')
            .filter(|x| !x.is_empty());

        (
            result.next().unwrap().to_string(),
            result.next().unwrap_or("").to_string(),
        )
    }

    pub async fn update_prompt(&mut self, new_prompt: String) {
        self.prompt = new_prompt.chars().take(self.max_size as usize).collect();
    }
}
