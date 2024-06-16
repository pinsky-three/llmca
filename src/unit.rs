use std::env;

use reqwest::header;
use serde_json::json;

use crate::api::ChatCompletionResponse;

#[derive(Default, Debug, Clone)]
pub struct CognitiveUnit {
    pub rule: String,
    pub state: Vec<String>,
    pub position: (usize, usize),
    // neighbors: Vec<(String, Vec<String>)>,
}

impl CognitiveUnit {
    pub fn calculate_next_state(&self, neighbors: Vec<(String, Vec<String>)>) -> String {
        self.llm_model_call(neighbors)
    }

    pub fn llm_model_call(&self, neighbors: Vec<(String, Vec<String>)>) -> String {
        let system_message = format!(
            "rule: {}
             state: {:?}
             neighbors: {:?}",
            self.rule, self.state, neighbors,
        );

        let input_message = "next_state:".to_string();

        let res = Self::generic_chat_completion(system_message, input_message);

        res.unwrap()
            .choices
            .first()
            .unwrap()
            .clone()
            .message
            .content
    }

    fn generic_chat_completion(
        system_message: String,
        user_message: String,
    ) -> Result<ChatCompletionResponse, Box<dyn std::error::Error>> {
        let open_ai_key = env::var("OPENAI_API_KEY").unwrap();
        let model_name = env::var("OPENAI_MODEL_NAME").unwrap_or("gpt-3.5-turbo".to_string());

        let mut headers = header::HeaderMap::new();

        headers.insert("Content-Type", "application/json".parse().unwrap());
        headers.insert(
            "Authorization",
            format!("Bearer {}", open_ai_key).parse().unwrap(),
        );

        let client = reqwest::blocking::Client::builder()
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

        let res = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .headers(headers)
            .body(body.to_string())
            .send()
            .unwrap()
            .json::<ChatCompletionResponse>()
            .unwrap();

        Ok(res)
    }
}
