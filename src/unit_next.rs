use chrono::{DateTime, Utc};

use reqwest::header;
use schemars::{schema_for, JsonSchema};

use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{api::ChatCompletionResponse, unit::CognitiveContext};

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
        // let cognitive_unit_description = CognitiveUnitWithMemory::self_description();
        let pair_description = CognitiveUnitPair::self_description();

        let system_message = [
            "You're a LLM Cognitive Unit and your unique task is to respond with your next CognitiveUnitPair based on the state of your neighbors in json format", 
            // format!("You have the following form: {}", cognitive_unit_description).as_str(),
            "If you rule is empty, you may to propose a new rule and your infer next CognitiveUnitPair",
            "The user pass to you your memory as a user input message list of CognitiveUnitPair in json format",
            format!("Always respond with a plain json of `CognitiveUnitPair`: {}", pair_description).as_str(),
            "Don't put the json in a code block, and don't add explanations, just the json ready to be parsed",
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
        neighbors: Vec<CognitiveUnitPair>,
    ) -> CognitiveUnitComplex {
        let input_payload = self
            .memory
            .iter()
            .map(|m| m.1.to_pair())
            .chain(neighbors.clone().into_iter())
            .map(|m| serde_json::to_string_pretty(&m).unwrap())
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

        let res_content = res_content
            .trim_matches(['`', '[', ']', '\n'])
            .trim_start_matches("json")
            .to_string();

        println!("res_content: {:?}", res_content);

        match serde_json::from_str::<CognitiveUnitPair>(&res_content) {
            Ok(output) => CognitiveUnitComplex {
                rule: output.rule,
                state: output.state,
                neighbors: neighbors.iter().map(|n| n.state.clone()).collect(),
                feedback: "".to_string(),
            },
            Err(err) => CognitiveUnitComplex {
                rule: self.memory.last().unwrap().1.rule.clone(),
                state: self.memory.last().unwrap().1.state.clone(),
                neighbors: neighbors.iter().map(|n| n.state.clone()).collect(),
                feedback: format!("Response Content: {}. Error: {}", res_content, err),
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
