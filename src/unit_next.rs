use chrono::{DateTime, Utc};

use reqwest::header;
use schemars::{schema_for, JsonSchema};

use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{api::ChatCompletionResponse, unit::CognitiveContext};
// Deserialize,
// use serde_json::json;

// use crate::api::ChatCompletionResponse;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CognitiveUnitComplex {
    pub rule: String,
    pub state: String,          // in json format
    pub neighbors: Vec<String>, // in json format
    pub feedback: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CognitiveUnitPair {
    pub rule: String,
    pub state: String,
}

impl CognitiveUnitPair {
    pub fn self_description() -> String {
        serde_json::to_string_pretty(&schema_for!(CognitiveUnitComplex)).unwrap()
    }
}

impl CognitiveUnitComplex {
    pub fn self_description() -> String {
        serde_json::to_string_pretty(&schema_for!(CognitiveUnitComplex)).unwrap()
    }

    pub fn to_pair(&self) -> CognitiveUnitPair {
        CognitiveUnitPair {
            rule: self.rule.clone(),
            state: self.state.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, JsonSchema)]
pub struct CognitiveUnitWithMemory {
    pub system_message: String,

    pub memory: Vec<(DateTime<Utc>, CognitiveUnitComplex)>,
    pub memory_size: usize,

    pub position: (usize, usize),
}

impl CognitiveUnitWithMemory {
    pub fn self_description() -> String {
        serde_json::to_string_pretty(&schema_for!(CognitiveUnitWithMemory)).unwrap()
    }

    pub fn new(
        position: (usize, usize),
        memory: Vec<(DateTime<Utc>, CognitiveUnitComplex)>,
        memory_size: usize,
    ) -> Self {
        let cognitive_unit_description = CognitiveUnitWithMemory::self_description();
        let pair_description = CognitiveUnitPair::self_description();

        let system_message = [
            "You're a LLM Cognitive Unit and your unique task is to respond with your next state based on the state of your neighbors in json format", 
            format!("You have the following form: {}", cognitive_unit_description).as_str(),
            format!("Return a `CognitiveUnitPair`: {}", pair_description).as_str(),
            "If you rule is empty, you may to propose a new rule and your infer next state",
            "The user pass to you your memory as a user input message list"
        ]
        .join(". ");

        Self {
            position,
            system_message,
            memory,
            memory_size,
        }
    }

    pub fn add_memory(&mut self, memory: (DateTime<Utc>, CognitiveUnitComplex)) {
        self.memory.push(memory);

        if self.memory.len() > self.memory_size {
            self.memory.remove(0);
        }
    }

    pub async fn calculate_next_complex(
        &self,
        ctx: &CognitiveContext,
        neighbors: Vec<CognitiveUnitComplex>,
    ) -> CognitiveUnitComplex {
        // let mut next_state = CognitiveUnitComplex {
        //     rule: "".to_string(),
        //     state: "".to_string(),
        //     neighbors: vec![],
        //     feedback: "".to_string(),
        // };

        // if !self.memory.is_empty() {
        //     let last_state = self.memory.last().unwrap().1.clone();
        //     next_state = last_state;
        // }

        // next_state

        // let input_payload = serde_json::to_string_pretty(&&CognitiveUnitInput {
        //     rule: self.rule.clone(),
        //     state: self.state.clone(),
        //     feedback: self.feedback.clone(),
        //     neighbors,
        // })
        // .unwrap();

        let input_payload = self
            .memory
            .iter()
            .map(|m| &m.1)
            .chain(neighbors.iter())
            .map(|m| serde_json::to_string_pretty(m).unwrap())
            .collect::<Vec<String>>();

        let res =
            Self::generic_chat_completion(ctx, self.system_message.clone(), input_payload).await;

        let res_content = res
            .unwrap()
            .choices
            .first()
            .unwrap()
            .clone()
            .message
            .content;

        println!("res_content: {:?}", res_content);

        match serde_json::from_str::<CognitiveUnitComplex>(&res_content) {
            Ok(output) => output,
            Err(err) => CognitiveUnitComplex {
                rule: "".to_string(),
                state: "".to_string(),
                neighbors: vec![],
                feedback: format!("Error: {}", err),
            },
        }
    }

    async fn generic_chat_completion(
        ctx: &CognitiveContext,
        system_message: String,
        user_messages: Vec<String>,
    ) -> Result<ChatCompletionResponse, Box<dyn std::error::Error>> {
        let mut headers = header::HeaderMap::new();

        headers.insert("Content-Type", "application/json".parse().unwrap());
        headers.insert(
            "Authorization",
            format!("Bearer {}", ctx.secret_key).parse().unwrap(),
        );

        let user_messages_json = user_messages
            .iter()
            .map(|m| json!({"role": "user", "content": m}))
            .collect::<Vec<Value>>();

        let mut messages = vec![json!({"role": "system", "content": system_message})];

        messages.extend_from_slice(&user_messages_json);

        let body = json!({
            "model": ctx.model_name,
            "messages": messages,
        });

        // println!("body: {:?}", body);

        let res = ctx
            .client
            .post(format!("{}/chat/completions", ctx.base_api))
            .headers(headers)
            .body(body.to_string())
            .send()
            .await
            .unwrap();

        // println!("res: {:?}", res);

        let parsed_res = res.json::<ChatCompletionResponse>().await.unwrap();

        Ok(parsed_res)
    }
}
