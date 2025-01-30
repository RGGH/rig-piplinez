use dotenv::dotenv;
use rig::{
    embeddings::EmbeddingsBuilder,
    parallel,
    pipeline::{self, agent_ops::lookup, passthrough, Op},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenv().ok();
    // Create OpenAI client
    let openai_client = Client::from_env();
    let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Create embeddings for our documents
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .document("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")?
        .document("Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")?
        .document("Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")?
        .build()
        .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings);

    // Create vector store index
    let index = vector_store.index(embedding_model);

    let agent = openai_client.agent("gpt-4")
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
        ")
        .build();

    let chain = pipeline::new()
        .chain(parallel!(
            passthrough(),
            lookup::<_, _, String>(index, 1), // Required to specify document type
        ))
        .map(|(prompt, maybe_docs)| match maybe_docs {
            Ok(docs) => format!(
                "Non standard word definitions:\n{}\n\n{}",
                docs.into_iter()
                    .map(|(_, _, doc)| doc)
                    .collect::<Vec<_>>()
                    .join("\n"),
                prompt,
            ),
            Err(err) => {
                println!("Error: {}! Prompting without additional context", err);
                format!("{prompt}")
            }
        })
        // Chain a "prompt" operation which will prompt out agent with the final prompt
        .prompt(agent);

    let response = chain.call("What does \"glarb-glarb\" mean?").await?;
    println!("{:?}", response);

    Ok(())
}
